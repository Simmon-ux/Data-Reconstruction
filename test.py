# test.py

"""
End-to-end TEST inference → HDF5 dumps (Antarctic SAT)

What this script does
- Loads a trained checkpoint (explicit --snapshot, or best/latest in --ckpt_dir).
- Builds the test dataset from HDF5:
    ./data/test_large/reanalysis_test.h5   (dataset 'tas')
    ./mask/mask_test.h5                    (dataset 'tas')
- Runs the model over the entire test split.
- Writes five HDF5 files under --out_prefix (dataset name: 'tas', shape [time,H,W]):
    • image.h5        = masked input  (gt * mask)
    • mask.h5         = binary mask
    • output.h5       = model prediction
    • output_comp.h5  = mask*input + (1-mask)*output
    • gt.h5           = ground truth (reanalysis)

Key args
- --root           Path to data root (default: ./data)
- --mask_test_h5   Path to test mask HDF5 (default: ./mask/mask_test.h5)
- --snapshot       Optional explicit checkpoint file (.pth)
- --ckpt_dir       Directory with checkpoints (fallback if --snapshot not given)
- --out_prefix     Output directory for HDF5 files (default: h5/)
- --batch_size     Inference batch size (default: 50)
- --n_threads      DataLoader workers (default: 0; Windows-safe)
- --input_channels Model input channels (default: 1 for SAT z-scores)

Usage
    python test.py \
      --root ./data \
      --mask_test_h5 ./mask/mask_test.h5 \
      --ckpt_dir ./snapshots/antarctica/ckpt \
      --out_prefix ./h5_test/ \
      --batch_size 50

Notes
- If --snapshot is not provided, the script looks for best.pth in --ckpt_dir,
  otherwise falls back to the most recent .pth file there.
- All tensors remain in normalized (z-score) space; no de-normalization is applied.
- Files are overwritten if present.
"""

import argparse, torch, os, glob
from places2 import Places2
from evaluation2 import evaluate2
from net import PConvUNet
from util.io import load_ckpt

def resolve_snapshot(args):
    # If user supplied a valid path, use it
    if args.snapshot and os.path.isfile(args.snapshot):
        return args.snapshot
    # Prefer best.pth in ckpt_dir
    best = os.path.join(args.ckpt_dir, 'best.pth')
    if os.path.isfile(best):
        return best
    # Otherwise, take the latest numbered checkpoint
    cands = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pth')))
    cands = [c for c in cands if os.path.basename(c) != 'best.pth']
    if not cands:
        raise FileNotFoundError(
            f'No checkpoints found in {args.ckpt_dir} and no --snapshot provided'
        )
    return cands[-1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='./data')
    p.add_argument('--mask_test_h5', type=str, default='./mask/mask_test.h5')
    p.add_argument('--snapshot', type=str, default='')  # optional now
    p.add_argument('--ckpt_dir', type=str, default='./snapshots/antarctica/ckpt')
    p.add_argument('--out_prefix', type=str, default='h5/')  # where to write HDF5
    p.add_argument('--batch_size', type=int, default=50)
    p.add_argument('--n_threads', type=int, default=0)  # Windows-safe
    p.add_argument('--input_channels', type=int, default=1)
    args = p.parse_args()

    ckpt_path = resolve_snapshot(args)
    print(f'Using checkpoint: {ckpt_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # We don't use transforms; Places2 returns [1,H,W] tensors already
    dataset_test = Places2(args.root, args.mask_test_h5, img_transform=None, mask_transform=None, split='test')

    model = PConvUNet(input_channels=args.input_channels).to(device)
    load_ckpt(ckpt_path, [('model', model)])
    model.eval()

    evaluate2(
        model, dataset_test, device,
        out_prefix=args.out_prefix,
        batch_size=args.batch_size,
        num_workers=args.n_threads
    )

if __name__ == '__main__':
    main()
