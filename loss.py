"""
Inpainting losses for Antarctic SAT reconstruction

What’s inside
- gram_matrix(feat):  Batch Gram matrix utility used by style loss (not active).
- total_variation_loss(image): TV regularizer (not active).
- InpaintingLoss(nn.Module): returns a dict of loss terms.

Active losses (used by training)
- 'hole'  : MSE over unobserved (mask==0) regions → drives reconstruction.
- 'valid' : MSE over observed (mask==1) regions   → preserves fidelity.

Inactive (commented) extras
- 'prc'   : Perceptual MSE using a feature extractor (e.g., VGG16).
- 'style' : Style loss via Gram matrices.
- 'tv'    : Total variation on the composite output.
Uncomment the block to enable; pass a valid feature extractor at init.

API
    loss_dict = InpaintingLoss(extractor=None)(input, mask, output, gt)

Args & shapes
- input  : [B,C,H,W] masked input to the model (typically gt * mask)
- mask   : [B,C,H,W] binary mask; 1 = observed, 0 = missing
- output : [B,C,H,W] model prediction
- gt     : [B,C,H,W] ground truth (reanalysis/target)
Returns
- dict with keys {'hole','valid'} (and optionally 'prc','style','tv' if enabled)

Notes
- Uses nn.MSELoss for all active terms.
- Composite used in optional blocks: output_comp = mask*input + (1-mask)*output
- Designed for C=1 (z-score SAT), but supports C=3 if you enable the perceptual/style branch.

Dependencies
- torch, numpy
"""


import torch
import torch.nn as nn
import numpy as np

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        #self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l2((1 - mask) * output, (1 - mask) * gt)

        loss_dict['valid'] = self.l2(mask * output, mask * gt)

        '''if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l2(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l2(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))
        
        loss_dict['tv'] = total_variation_loss(output_comp)'''

        return loss_dict
#-----------------------------------------------------------------------------------------------------------------