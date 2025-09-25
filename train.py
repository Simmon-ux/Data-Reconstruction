# train.py

"""
Training loop for Antarctic SAT inpainting (PConv U-Net)

What this script does
- Builds train/val datasets from HDF5:
    ./data/train_large/reanalysis_train.h5  + ./mask/mask_train.h5
    ./data/val_large/reanalysis_val.h5      + ./mask/mask_val.h5
- Trains a compact Partial Convolution U-Net in z-score space.
- Logs to TensorBoard, periodically validates, saves checkpoints, and
  writes quick visual grids of predictions.

Key features
- InfiniteSampler for continuous epoch-less training.
- Loss: InpaintingLoss (MSE on valid + hole regions; weighted via opt.LAMBDA_DICT).
- Validation metric: full-field MSE(model_output, gt).
- Checkpoints:
    • {save_dir}/ckpt/best.pth (best val MSE)
    • {save_dir}/ckpt/{iter}.pth (periodic snapshots)
- Optional fine-tuning flag freezes encoder BN and lowers LR.

Important arguments
- --root                Data root (default: ./data)
- --mask_train_h5       Train mask H5 (default: ./mask/mask_train.h5)
- --mask_val_h5         Val mask H5   (default: ./mask/mask_val.h5)
- --max_iter            Total iterations (default: 9000)
- --batch_size          Batch size (default: 50)
- --save_model_interval Save ckpt every N iters (default: 200)
- --val_interval        Validate every N iters (default: 200)
- --log_interval        TB log every N iters (default: 50)
- --lr / --lr_finetune  Base / fine-tune learning rate
- --save_dir            Output dir for snapshots (default: ./snapshots/antarctica)
- --log_dir             TensorBoard logs (default: ./logs/antarctica)
- --resume              Path to checkpoint to resume (optional)
- --finetune            Enable fine-tune mode (freeze encoder BN)
- --patience            Early-stopping patience in validation steps (default: 25)
- --input_channels      Model input channels (1 for SAT z-scores)

Outputs
- Checkpoints under {save_dir}/ckpt/
- TensorBoard under {log_dir}/
- Periodic preview images under {save_dir}/images/

Dependencies
- torch, torchvision, tensorboardX, numpy, tqdm, h5py (via dataset)
"""

import argparse, os, numpy as np, torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from places2 import Places2
from util.io import load_ckpt, save_ckpt

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, n): self.n = n
    def __iter__(self): return iter(self.loop())
    def __len__(self): return 2**31
    def loop(self):
        i = 0; order = np.random.permutation(self.n)
        while True:
            yield order[i]; i += 1
            if i >= self.n:
                np.random.seed(); order = np.random.permutation(self.n); i = 0

def build_parser():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--root', type=str, default='./data')
    p.add_argument('--mask_train_h5', type=str, default='./mask/mask_train.h5')
    p.add_argument('--mask_val_h5',   type=str, default='./mask/mask_val.h5')
    # Paper schedule
    p.add_argument('--max_iter', type=int, default=9000)
    p.add_argument('--batch_size', type=int, default=50)
    p.add_argument('--save_model_interval', type=int, default=200)
    p.add_argument('--val_interval', type=int, default=200)
    p.add_argument('--log_interval', type=int, default=50)
    # Opt/infra
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr_finetune', type=float, default=5e-5)
    p.add_argument('--n_threads', type=int, default=0)  # ← default 0 for Windows
    p.add_argument('--save_dir', type=str, default='./snapshots/antarctica')
    p.add_argument('--log_dir', type=str, default='./logs/antarctica')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--finetune', action='store_true')
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--input_channels', type=int, default=1)  # using 1-ch
    return p

def main():
    args = build_parser().parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(f'{args.save_dir}/images', exist_ok=True)
    os.makedirs(f'{args.save_dir}/ckpt', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    img_tf  = transforms.Compose([])
    mask_tf = transforms.Compose([])

    dataset_train = Places2(args.root, args.mask_train_h5, img_tf, mask_tf, 'train')
    dataset_val   = Places2(args.root, args.mask_val_h5,   img_tf, mask_tf, 'val')

    iterator_train = iter(data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=InfiniteSampler(len(dataset_train)),
        num_workers=args.n_threads, pin_memory=False))

    val_loader = data.DataLoader(dataset_val, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.n_threads, pin_memory=False)

    model = PConvUNet(input_channels=args.input_channels).to(device)
    lr = args.lr_finetune if args.finetune else args.lr
    if args.finetune: model.freeze_enc_bn = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = InpaintingLoss(extractor=None).to(device)   # pure MSE
    crit_mse_full = torch.nn.MSELoss(reduction='mean')

    start_iter = 0
    if args.resume:
        start_iter = load_ckpt(args.resume, [('model', model)], [('optimizer', optimizer)])
        for pg in optimizer.param_groups: pg['lr'] = lr
        print('Resumed from', start_iter)

    best_val = float('inf'); bad = 0

    def compute_val_mse():
        model.eval()
        total = 0.0; n = 0
        with torch.no_grad():
            for image, mask, gt in val_loader:
                image, mask, gt = image.to(device), mask.to(device), gt.to(device)
                out, _ = model(image, mask)
                mse = crit_mse_full(out, gt).item()
                total += mse * image.size(0); n += image.size(0)
        return total / max(1, n)

    for it in tqdm(range(start_iter, args.max_iter), total=args.max_iter-start_iter):
        model.train()
        image, mask, gt = [x.to(device) for x in next(iterator_train)]
        out, _ = model(image, mask)
        loss_dict = criterion(image, mask, out, gt)

        loss = 0.0
        for k, w in opt.LAMBDA_DICT.items():   # {'valid':1.0, 'hole':6.0}
            v = w * loss_dict[k]; loss += v
            if (it + 1) % args.log_interval == 0:
                writer.add_scalar(f'train/{k}', v.item(), it + 1)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (it + 1) % args.save_model_interval == 0 or (it + 1) == args.max_iter:
            save_ckpt(f'{args.save_dir}/ckpt/{it + 1}.pth',
                      [('model', model)], [('optimizer', optimizer)], it + 1)

        if (it + 1) % args.val_interval == 0 or (it + 1) == args.max_iter:
            val_mse = compute_val_mse()
            writer.add_scalar('val/mse_full', val_mse, it + 1)
            if val_mse + 1e-7 < best_val:
                best_val = val_mse; bad = 0
                save_ckpt(f'{args.save_dir}/ckpt/best.pth',
                          [('model', model)], [('optimizer', optimizer)], it + 1)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'Early stopping at iter {it+1}; best val MSE={best_val:.6f}')
                    break

        if (it + 1) % (5 * args.val_interval) == 0:
            model.eval()
            evaluate(model, dataset_val, device, f'{args.save_dir}/images/val_{it + 1}.jpg')

    writer.close()

if __name__ == '__main__':
    main()
