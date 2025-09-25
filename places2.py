# places2.py

"""
HDF5-backed Dataset for Antarctic SAT inpainting

What it loads
- Reanalysis (normalized, z-score) from split-specific HDF5:
    ./data/train_large/reanalysis_train.h5
    ./data/val_large/reanalysis_val.h5
    ./data/test_large/reanalysis_test.h5
  (dataset name: 'tas' with shape [time,H,W])
- Observation-derived binary mask from a separate HDF5 (dataset 'tas', shape [time,H,W]).

What it returns (per __getitem__)
- image_masked : [1,H,W] = gt * mask         (what the model sees)
- mask         : [1,H,W] ∈ {0,1}             (1=observed, 0=missing)
- gt           : [1,H,W]                      (full reanalysis target, z-score space)

Key behaviors
- Opens HDF5 files on demand inside __getitem__ (safe for multi-worker DataLoader).
- Infers dataset length/shape once during __init__ without keeping file handles open.
- No transforms applied by default; placeholders `img_transform` / `mask_transform` are accepted
  but unused (kept for API compatibility).

Usage
    ds = Places2(img_root='./data', mask_h5='./data/mask/mask_train.h5',
                 img_transform=None, mask_transform=None, split='train')
    img, m, gt = ds[0]   # each is a float32 torch tensor [1,H,W]

Assumptions
- Values are already normalized (z-scores).
- The mask file is time-aligned with the chosen split (same [time,H,W]).
- HDF5 dataset name is 'tas' for both reanalysis and mask files.

Dependencies
- h5py, numpy, torch
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data

class Places2(data.Dataset):
    """
    Returns (image_masked, mask, gt) as float32 tensors [1,H,W]
      gt   = full reanalysis (z-score)
      mask = observation-derived binary mask (0/1)
      image_masked = gt * mask
    """
    def __init__(self, img_root: str, mask_h5: str, img_transform, mask_transform, split='train'):
        super().__init__()
        if split == 'train':
            self.rean_path = f'{img_root}/train_large/reanalysis_train.h5'
        elif split == 'val':
            self.rean_path = f'{img_root}/val_large/reanalysis_val.h5'
        elif split == 'test':
            self.rean_path = f'{img_root}/test_large/reanalysis_test.h5'
        else:
            raise ValueError(f'Unknown split: {split}')
        self.mask_path = mask_h5
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # Read shapes ONLY; do NOT keep any h5py objects on self
        with h5py.File(self.rean_path, 'r') as h5:
            tas = h5['tas']
            self.length = tas.shape[0]
            self.H, self.W = tas.shape[1], tas.shape[2]

    def __len__(self): 
        return self.length

    def __getitem__(self, index):
        # Open files on demand (safe for multi-worker DataLoader)
        with h5py.File(self.rean_path, 'r') as h5:
            gt2d = h5['tas'][index, :, :].astype(np.float32)
        with h5py.File(self.mask_path, 'r') as hm:
            m2d = hm['tas'][index, :, :].astype(np.float32)  # 0/1

        gt   = torch.from_numpy(gt2d).unsqueeze(0)   # [1,H,W]
        mask = torch.from_numpy(m2d).unsqueeze(0)    # [1,H,W]
        image_masked = gt * mask
        return image_masked, mask, gt
