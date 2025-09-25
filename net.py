"""
PConv U-Net for Antarctic SAT inpainting (compact variant)

What’s inside
- weights_init(init_type): Common initializers for Conv/Linear (gaussian/xavier/kaiming/orthogonal).
- VGG16FeatureExtractor: Optional feature pyramid (no pretrained weights) for perceptual/style loss.
- PartialConv: Mask-aware convolution (per NVIDIA PConv), propagates an updated mask.
- PCBActiv: PartialConv + optional BatchNorm + activation; supports downsampling variants.
- PConvUNet: Compact encoder–decoder with PConv blocks and skip connections.

Key details / fixes
- Decoder upsampling targets the **exact** spatial size of each skip feature to avoid rounding mismatches
  (no 64→127 off-by-one artifacts).
- PartialConv re-normalizes outputs by the number of valid inputs in the receptive field and zeros out
  locations with no valid coverage (mask_sum == 0), returning a new binary mask.
- Encoder BN layers can be frozen during fine-tuning by setting `model.freeze_enc_bn = True`.

Shapes & API
- Forward signature: `y, y_mask = model(input, input_mask)`
  * input:     [B, C, H, W] (C=1 for SAT z-scores, or 3 if you extend to multi-channel)
  * input_mask:[B, C, H, W] (1=observed, 0=missing); same C as input
  * y:         [B, C, H, W] (prediction in normalized space)
  * y_mask:    [B, C, H, W] (propagated valid mask after decoding)
- Default architecture (layer_size=3) with channels: C→18→36→72 (encoder), mirror in decoder.

Usage (example)
    model = PConvUNet(input_channels=1).to(device)
    out, out_mask = model(x, m)  # x,m: [B,1,H,W]

Dependencies
- torch, torchvision (for optional VGG backbone)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ------------------------------
# Initializers
# ------------------------------
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


# ------------------------------
# VGG16 Feature Extractor
# (we do NOT download weights; you’re using pure MSE)
# ------------------------------
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Avoid downloading weights (we only need structure if ever used)
        try:
            vgg16 = models.vgg16(weights=None)      # torchvision ≥ 0.13
        except TypeError:
            vgg16 = models.vgg16(pretrained=False)  # older torchvision
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # freeze
        for i in range(3):
            for p in getattr(self, f'enc_{i+1}').parameters():
                p.requires_grad = False

    def forward(self, image):
        # returns 3 feature maps
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1}')
            results.append(func(results[-1]))
        return results[1:]


# ------------------------------
# Partial Convolution (mask-propagating)
# ------------------------------
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        # initialize mask conv to ones and keep it fixed
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # Based on NVIDIA partial conv:
        # https://arxiv.org/abs/1804.07723
        with torch.no_grad():
            # mask sum in each receptive field
            mask_sum = self.mask_conv(mask)
        # normal conv on masked input
        output_pre = self.input_conv(input * mask)

        # valid locations are where at least one input was observed
        no_update_holes = (mask_sum == 0)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        # re-normalize by mask coverage
        output = output_pre * self.mask_conv.weight.numel() / mask_sum
        output = output.masked_fill_(no_update_holes, 0.0)

        # new binary mask
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        return output, new_mask


# ------------------------------
# PartialConv + BN + Activation convenience block
# ------------------------------
class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu', conv_bias=False):
        super().__init__()
        # downsample choices used by original PConv U-Net
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)

        if activ == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activ is None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activ: {activ}')

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if getattr(self, 'activation', None) is not None:
            h = self.activation(h)
        return h, h_mask


# ------------------------------
# PConv U-Net (compact variant you’re using)
# Changes:
#  - Decoder now upsamples to the EXACT shape of the skip feature,
#    avoiding 64->127 hacks and scale_factor rounding issues.
# ------------------------------
class PConvUNet(nn.Module):
    def __init__(self, layer_size=3, input_channels=3, upsampling_mode='nearest'):
        """
        Compact PConv U-Net:
          enc_1: C -> 18 (down-7)
          enc_2: 18 -> 36 (down-5)
          enc_3: 36 -> 72 (down-5)
          enc_4: 72 -> 72 (down-3)
          dec_4..dec_1 mirror with skip concatenations
        """
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size

        # Encoder
        self.enc_1 = PCBActiv(input_channels, 18, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(18, 36, sample='down-5')
        self.enc_3 = PCBActiv(36, 72, sample='down-5')
        self.enc_4 = PCBActiv(72, 72, sample='down-3')
        for i in range(4, self.layer_size):
            name = f'enc_{i+1}'
            setattr(self, name, PCBActiv(72, 72, sample='down-3'))

        # Decoder
        for i in range(4, self.layer_size):
            name = f'dec_{i+1}'
            setattr(self, name, PCBActiv(72 + 72, 72, activ='leaky'))
        self.dec_4 = PCBActiv(72 + 72, 72, activ='leaky')
        self.dec_3 = PCBActiv(72 + 36, 36, activ='leaky')
        self.dec_2 = PCBActiv(36 + 18, 18, activ='leaky')
        self.dec_1 = PCBActiv(18 + input_channels, input_channels, bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}       # encoder features
        h_mask_dict = {}  # encoder masks

        # store input
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        # encoder path
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):  # 1..layer_size
            l_key = f'enc_{i}'
            h_key = f'h_{i}'
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        # start from the deepest encoder feature
        h_key = f'h_{self.layer_size}'
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # decoder with skip connections
        for i in range(self.layer_size, 0, -1):  # e.g., 3,2,1
            enc_h_key = f'h_{i-1}'
            dec_l_key = f'dec_{i}'

            # === KEY FIX: upsample to EXACT encoder spatial size ===
            target_hw = h_dict[enc_h_key].shape[-2:]        # (H, W) of skip feature
            h      = F.interpolate(h,      size=target_hw, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, size=target_hw, mode='nearest')

            # concat skip (feature & mask), then decode
            h      = torch.cat([h,      h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """Freeze encoder BN layers when requested."""
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


# Debug (optional)
if __name__ == '__main__':
    size = (1, 3, 180, 180)
    x = torch.randn(size)
    m = torch.ones_like(x)
    model = PConvUNet()
    y, ym = model(x, m)
    print(y.shape, ym.shape)
