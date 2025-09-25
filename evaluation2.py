# evaluation2.py

"""
Full-dataset inference → HDF5 exports (Antarctic SAT)

What it does
- Runs the model over the ENTIRE dataset (no shuffling).
- Saves five HDF5 files (dataset name: 'tas', shape [time,H,W]):
    • image.h5        = masked input  (gt * mask)
    • mask.h5         = binary mask
    • output.h5       = model prediction
    • output_comp.h5  = mask*input + (1-mask)*output
    • gt.h5           = ground truth (reanalysis)
- HDF5 dims are labeled: time / lat / lon.

API
    evaluate2(model, dataset, device,
              out_prefix='h5/', batch_size=16, num_workers=0)

Args
- model      : network with forward(image, mask) → (output, _)
- dataset    : yields (image_masked, mask, gt) each [C,H,W] (C=1 or 3)
- device     : torch.device
- out_prefix : output folder to place the .h5 files
- batch_size : inference batch size
- num_workers: DataLoader workers

Notes
- Works with both 1-channel and 3-channel tensors; saves the first (or middle) channel as 2D grids.
- All values remain in normalized (z-score) space; no de-normalization.
- Files are overwritten if they exist.

Dependencies
- torch, h5py
"""

import os
import h5py
import torch
from torch.utils.data import DataLoader

def _save_h5(path, key, tensor_3d):
    """
    Save a [T,H,W] tensor to HDF5 with dataset name `key`
    and labeled dims (time, lat, lon).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = tensor_3d.numpy().astype('float32')
    with h5py.File(path, 'w') as h:
        d = h.create_dataset(key, data=arr, dtype='f4')
        d.dims[0].label = 'time'
        d.dims[1].label = 'lat'
        d.dims[2].label = 'lon'

def evaluate2(model, dataset, device, out_prefix='h5/', batch_size=16, num_workers=0):
    """
    Run inference over the ENTIRE dataset and export:
      - image.h5       (masked input)
      - mask.h5
      - output.h5      (model prediction)
      - output_comp.h5 (mask*input + (1-mask)*output)
      - gt.h5          (full reanalysis target)
    Each file contains dataset 'tas' with shape [time, H, W].
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    imgs, masks, gts, outs, outs_comp = [], [], [], [], []

    with torch.no_grad():
        for image, mask, gt in loader:
            # image/mask/gt are [B, C, H, W]; C is 1 (your setup) or 3 (legacy)
            C = image.shape[1]
            ch = 0 if C == 1 else 1  # use first channel for 1-ch; middle for 3-ch

            image = image.to(device)
            mask  = mask.to(device)
            output, _ = model(image, mask)
            output_comp = mask * image + (1 - mask) * output

            # Select the channel to save, move to CPU
            imgs.append(image[:, ch, :, :].cpu())
            masks.append(mask[:, ch, :, :].cpu())
            gts.append(gt[:, ch, :, :].cpu())
            outs.append(output[:, ch, :, :].cpu())
            outs_comp.append(output_comp[:, ch, :, :].cpu())

    imgs       = torch.cat(imgs,      dim=0)
    masks      = torch.cat(masks,     dim=0)
    gts        = torch.cat(gts,       dim=0)
    outs       = torch.cat(outs,      dim=0)
    outs_comp  = torch.cat(outs_comp, dim=0)

    os.makedirs(out_prefix, exist_ok=True)
    _save_h5(os.path.join(out_prefix, 'image.h5'),       'tas', imgs)
    _save_h5(os.path.join(out_prefix, 'mask.h5'),        'tas', masks)
    _save_h5(os.path.join(out_prefix, 'output.h5'),      'tas', outs)
    _save_h5(os.path.join(out_prefix, 'output_comp.h5'), 'tas', outs_comp)
    _save_h5(os.path.join(out_prefix, 'gt.h5'),          'tas', gts)
