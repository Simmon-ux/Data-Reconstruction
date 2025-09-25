
"""
Antarctic SAT • Panel C: Reconstruction + Validation + Outputs (end-to-end)

What this script does
- Builds an observation-driven dataset with an optional HOLD-OUT subset of obs
  (withheld from model input, kept for scoring).
- Trains (or resumes) your DLM (via train.py) until validation gates pass:
    Gate-1: MSE on HOLD-OUT observations
    Gate-2:  trend agreement on HOLD-OUT (Pearson r + sign agreement)
- After passing gates, reconstructs the FULL daily field (all observations),
  and exports:
    • recon_out/recon.h5, recon_comp.h5, obs_mask.h5  (HDF5, dataset 'tas')
    • recon_out/Antarctic_SAT_EASE_daily_1979_2023.nc
    • recon_out/Antarctic_SAT_EASE_monthly_1979_2023.nc
    •  1×1° daily & monthly:
         Antarctic_SAT_1deg_daily_1979_2023.nc
         Antarctic_SAT_1deg_monthly_1979_2023.nc

Inputs expected
- ./data/obs/observations_full.h5   (dataset 'tas': [T,H,W], normalized/z-score space)
- Model code in repo: net.py (PConvUNet), train.py, evaluation2.py (_save_h5), util/io.py
- (For 1×1°) 2-D EASE lat/lon arrays (H×W) supplied via NPZ (keys: lat, lon) or NetCDF.

Key args
- --mse_threshold <float>        Gate-1 threshold (default 1.50)
- --do_trend_gate                Enable Gate-2 (trend validation)
- --trend_min_days <int>         Min hold-out coverage per pixel (~10y default 3650)
- --trend_min_r <float>          Min Pearson r for trends (default 0.35)
- --trend_min_sign_agree <float> Min fraction of pixels with matching trend sign (default 0.60)
- --val_holdout_frac <float>     Fraction of obs withheld for validation (default 0.10)
- --make_1deg                    Also write 1×1° daily/monthly NetCDF (no xESMF)
- --ease_latlon <path>           NPZ/NC file with 2-D EASE lat/lon (required if --make_1deg)

Usage
    # 1) Ensure ./data/obs/observations_full.h5 exists (from nc_to_h5.py)
    # 2) Run the pipeline (loops training until gates pass):
    python run_workflow_1deg_no_xesmf.py \
        --mse_threshold 1.50 \
        --val_holdout_frac 0.10 \
        --do_trend_gate \
        --trend_min_r 0.35 \
        --trend_min_sign_agree 0.60 \
        --make_1deg \
        --ease_latlon data/ease_latlon_180x180.npz

Notes
- Values remain in normalized (z-score) space; de-normalize downstream if needed.
- Hold-out is time-varying and reproducible via --val_holdout_seed.
- 1×1° step uses a fast bin-average/nearest approach (no xESMF dependency).
- Files are overwritten if present; outputs live under ./recon_out/.
"""

import os
import argparse
import subprocess
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from net import PConvUNet
from util.io import load_ckpt
from evaluation2 import _save_h5  # re-use your HDF5 writer


# ============================================================
# Panel C: Reconstruction • Validation • Outputs
# ============================================================

class ObsDataset(Dataset):
    """
    Observation-driven dataset with validation hold-out support.

    If val_holdout_frac > 0, a random subset of observed pixels are WITHHELD
    from the model input (train_mask=0) but kept for scoring (hold_mask=1).

    Returns (per item):
      image_masked : [1,H,W] = gt_obs * train_mask   (what the model sees)
      train_mask   : [1,H,W]
      gt_obs       : [1,H,W]   (zeros where missing)
      hold_mask    : [1,H,W]   (1 on WITHHELD obs used for validation)
    """
    def __init__(self,
                 obs_h5: str = './data/obs/observations_full.h5',
                 val_holdout_frac: float = 0.0,
                 val_holdout_seed: int = 42):
        super().__init__()
        with h5py.File(obs_h5, 'r') as h:
            arr = h['tas'][:]  # [T,H,W], float32, may contain NaNs
        self.T, self.H, self.W = arr.shape

        full_mask = np.isfinite(arr).astype('float32')
        gt        = np.nan_to_num(arr, nan=0.0).astype('float32')

        rng = np.random.RandomState(val_holdout_seed)
        hold = np.zeros_like(full_mask, dtype='float32')
        if val_holdout_frac > 0:
            for t in range(self.T):
                idx = np.where(full_mask[t].reshape(-1) > 0)[0]
                k = int(round(val_holdout_frac * idx.size))
                if k > 0:
                    pick = rng.choice(idx, size=k, replace=False)
                    hold[t].reshape(-1)[pick] = 1.0

        train_mask = full_mask * (1.0 - hold)

        self.gt         = gt
        self.full_mask  = full_mask
        self.train_mask = train_mask
        self.hold_mask  = hold

    def __len__(self): return self.T

    def __getitem__(self, i):
        gt2d     = torch.from_numpy(self.gt[i]).unsqueeze(0)          # [1,H,W]
        m_train  = torch.from_numpy(self.train_mask[i]).unsqueeze(0)
        m_hold   = torch.from_numpy(self.hold_mask[i]).unsqueeze(0)
        image    = gt2d * m_train
        return image, m_train, gt2d, m_hold


@torch.no_grad()
def mse_on_holdout(model, loader, device):
    """
    Mean squared error computed ONLY on hold-out observation pixels.
    """
    crit = torch.nn.MSELoss(reduction='sum')
    total_err = 0.0
    total_cnt = 0.0
    for image, mtrain, gt, mhold in loader:
        image, mtrain, gt, mhold = image.to(device), mtrain.to(device), gt.to(device), mhold.to(device)
        out, _ = model(image, mtrain)
        resid = (out - gt) * mhold
        total_err += crit(resid, torch.zeros_like(resid)).item()
        total_cnt += mhold.sum().item()
    return float('inf') if total_cnt == 0 else (total_err / total_cnt)


def _slope_per_day(t_days: np.ndarray, y: np.ndarray) -> float:
    a, b = np.polyfit(t_days.astype(float), y.astype(float), 1)
    return float(a)


def trend_gate_on_holdout(recon: np.ndarray,
                          obs: np.ndarray,
                          hold_mask: np.ndarray,
                          min_days: int = 3650) -> tuple[float, float]:
    
    T, H, W = recon.shape
    t = np.arange(T, dtype=float)
    tr_r, tr_o, agree = [], [], []

    for i in range(H):
        mrow = hold_mask[:, i, :] > 0.5
        cnt  = mrow.sum(axis=0)
        cols = np.where(cnt >= min_days)[0]
        if cols.size == 0:
            continue
        for j in cols:
            idx = mrow[:, j]
            sr = _slope_per_day(t[idx], recon[idx, i, j]) * 3652.5
            so = _slope_per_day(t[idx], obs[idx, i, j])   * 3652.5
            tr_r.append(sr); tr_o.append(so)
            agree.append(np.sign(sr) == np.sign(so))

    if len(tr_r) < 5:
        return 0.0, 0.0
    r = np.corrcoef(np.asarray(tr_r), np.asarray(tr_o))[0, 1]
    return float(r), float(np.mean(agree))


@torch.no_grad()
def reconstruct_for_validation(model, ds_val, device):
    ld = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=0)
    outs, holds = [], []
    for image, mtrain, _, mhold in ld:
        image, mtrain = image.to(device), mtrain.to(device)
        out, _ = model(image, mtrain)
        outs.append(out[:, 0].cpu()); holds.append(mhold[:, 0].cpu())
    return torch.cat(outs).numpy(), torch.cat(holds).numpy()


@torch.no_grad()
def reconstruct_and_save(model, ds_final, device, out_dir='recon_out'):
    """
    Final reconstruction with ALL observations (no hold-out).
    """
    os.makedirs(out_dir, exist_ok=True)
    ld = DataLoader(ds_final, batch_size=32, shuffle=False, num_workers=0)
    outs, comps, masks = [], [], []
    for image, mtrain, _, _ in ld:
        image, mtrain = image.to(device), mtrain.to(device)
        out, _ = model(image, mtrain)
        comp = mtrain * image + (1 - mtrain) * out
        outs.append(out[:, 0].cpu()); comps.append(comp[:, 0].cpu()); masks.append(mtrain[:, 0].cpu())
    outs = torch.cat(outs); comps = torch.cat(comps); masks = torch.cat(masks)
    _save_h5(os.path.join(out_dir, 'recon.h5'),      'tas', outs)
    _save_h5(os.path.join(out_dir, 'recon_comp.h5'), 'tas', comps)
    _save_h5(os.path.join(out_dir, 'obs_mask.h5'),   'tas', masks)
    return outs.numpy()


def save_monthlies_ease(recon_daily: np.ndarray, out_dir='recon_out'):
    """
    EASE daily & monthly NetCDF exports.
    """
    import xarray as xr
    T, H, W = recon_daily.shape
    time = np.arange(T, dtype='timedelta64[D]') + np.datetime64('1979-01-01')
    ds = xr.Dataset({'SAT': (('time','y','x'), recon_daily.astype('float32'))},
                    coords={'time': time, 'y': np.arange(H), 'x': np.arange(W)})
    ds_m = ds.resample(time='1MS').mean()
    ds.to_netcdf(os.path.join(out_dir, 'Antarctic_SAT_EASE_daily_1979_2023.nc'))
    ds_m.to_netcdf(os.path.join(out_dir, 'Antarctic_SAT_EASE_monthly_1979_2023.nc'))


# ============================================================
# 1°×1° DAILY & MONTHLY without xESMF (bin-average / nearest)
# ============================================================

def _load_ease_latlon(latlon_path: str):
    """
    Load 2-D EASE lat/lon arrays (shape [H,W]) from NPZ ('lat','lon') or NetCDF.
    """
    import xarray as xr
    if latlon_path.lower().endswith('.npz'):
        z = np.load(latlon_path)
        lat2d = z['lat']; lon2d = z['lon']
    else:
        ds = xr.open_dataset(latlon_path)
        lat_name = [v for v in list(ds.data_vars) + list(ds.coords) if str(v).lower().startswith('lat')][0]
        lon_name = [v for v in list(ds.data_vars) + list(ds.coords) if str(v).lower().startswith('lon')][0]
        lat2d = ds[lat_name].values
        lon2d = ds[lon_name].values
    return lat2d.astype('float64'), lon2d.astype('float64')


def _prep_1deg_bins(lat2d: np.ndarray, lon2d: np.ndarray,
                    pad_deg: float = 0.0):
    """
    Build 1° lat/lon bin edges/vectors and a per-pixel linear bin index map.

    Returns:
      lat_vec [nlat], lon_vec [nlon], bin_index [H,W] (linear index into nlat*nlon, -1 if out)
    """
    # Normalize lon to [-180, 180)
    lon2d = ((lon2d + 180.) % 360.) - 180.

    lat_min = np.floor(np.nanmin(lat2d) - pad_deg)
    lat_max = np.ceil (np.nanmax(lat2d) + pad_deg)
    lon_min = np.floor(np.nanmin(lon2d) - pad_deg)
    lon_max = np.ceil (np.nanmax(lon2d) + pad_deg)

    lat_vec = np.arange(lat_min, lat_max + 1e-6, 1.0)  # centers
    lon_vec = np.arange(lon_min, lon_max + 1e-6, 1.0)

    # Compute index per EASE pixel into the 1° grid
    # Centers → bin numbers (round to nearest degree bin)
    lat_idx = np.round(lat2d - lat_vec[0]).astype(int)
    lon_idx = np.round(lon2d - lon_vec[0]).astype(int)

    H, W = lat2d.shape
    nlat, nlon = len(lat_vec), len(lon_vec)
    bin_index = -np.ones((H, W), dtype=np.int64)
    inside = (lat_idx >= 0) & (lat_idx < nlat) & (lon_idx >= 0) & (lon_idx < nlon)
    lin = lat_idx * nlon + lon_idx
    bin_index[inside] = lin[inside]
    return lat_vec.astype('float64'), lon_vec.astype('float64'), bin_index


def _create_daily_1deg_netcdf(path, T, lat_vec, lon_vec):
    """
    Create a NetCDF file with unlimited 'time' and variables:
      SAT(time, lat, lon)
    """
    from netCDF4 import Dataset
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nc = Dataset(path, 'w')
    nc.createDimension('time', T)
    nc.createDimension('lat', len(lat_vec))
    nc.createDimension('lon', len(lon_vec))

    vtime = nc.createVariable('time', 'i4', ('time',))
    vlat  = nc.createVariable('lat',  'f4', ('lat',))
    vlon  = nc.createVariable('lon',  'f4', ('lon',))
    vsat  = nc.createVariable('SAT',  'f4', ('time','lat','lon'), zlib=True, complevel=4, fill_value=np.nan)

    vlat[:] = lat_vec.astype('float32')
    vlon[:] = lon_vec.astype('float32')
    vtime[:] = np.arange(T, dtype=np.int32)  # relative index; can post-process to CF time

    nc.description = 'Antarctic SAT 1x1 deg (daily), z-score space, bin-averaged from EASE grid'
    nc.history     = 'Generated by run_workflow_1deg_no_xesmf.py'
    return nc, vsat


def _bin_average_to_1deg(day2d: np.ndarray, bin_index: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    """
    Bin-average one EASE 2D field onto 1° lattice using precomputed bin_index.
    """
    flat_vals = day2d.ravel()
    flat_bins = bin_index.ravel()
    valid = np.isfinite(flat_vals) & (flat_bins >= 0)

    if not np.any(valid):
        return np.full((nlat, nlon), np.nan, dtype=np.float32)

    idx = flat_bins[valid].astype(np.int64)
    vals = flat_vals[valid].astype(np.float64)

    sums = np.bincount(idx, weights=vals, minlength=nlat*nlon)
    cnts = np.bincount(idx, weights=np.ones_like(vals), minlength=nlat*nlon)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / cnts
    out = means.reshape(nlat, nlon).astype('float32')
    out[cnts.reshape(nlat, nlon) == 0] = np.nan
    return out


def make_1deg_products_without_xesmf(ease_daily_nc: str,
                                     ease_latlon_path: str,
                                     out_dir: str = 'recon_out',
                                     chunk: int = 64,
                                     method: str = 'bin'):  # 'bin' or 'nearest' (bin uses nearest-to-center anyway)
    
    import xarray as xr
    from netCDF4 import Dataset

    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(ease_daily_nc)
    var = 'SAT' if 'SAT' in ds else list(ds.data_vars)[0]
    T   = ds.dims['time']
    H   = ds.dims['y']; W = ds.dims['x']

    # Load lat/lon and precompute bin map
    lat2d, lon2d = _load_ease_latlon(ease_latlon_path)
    assert lat2d.shape == (H, W) and lon2d.shape == (H, W), \
        f'Lat/Lon shape {lat2d.shape}/{lon2d.shape} must match EASE dims {(H,W)}'
    lat_vec, lon_vec, bin_index = _prep_1deg_bins(lat2d, lon2d)
    nlat, nlon = len(lat_vec), len(lon_vec)

    # Create DAILY 1deg file
    daily_path = os.path.join(out_dir, 'Antarctic_SAT_1deg_daily_1979_2023.nc')
    nc, vsat = _create_daily_1deg_netcdf(daily_path, T, lat_vec, lon_vec)

    # Process in chunks to limit RAM
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        # Load chunk [t0:t1] into memory (as numpy)
        chunk_vals = ds[var].isel(time=slice(t0, t1)).values.astype('float32')  # [B,H,W]
        # Map each day to 1deg
        out_block = np.empty((t1 - t0, nlat, nlon), dtype='float32')
        for k in range(t1 - t0):
            out_block[k] = _bin_average_to_1deg(chunk_vals[k], bin_index, nlat, nlon)
        # Write block
        vsat[t0:t1, :, :] = out_block
        print(f'[1deg] wrote days {t0}..{t1-1}')

    # Close daily file
    nc.close()

    # Make MONTHLY 1deg by resampling the daily file we just wrote
    ds1 = xr.open_dataset(daily_path)
    ds1_m = ds1.resample(time='1MS').mean()
    monthly_path = os.path.join(out_dir, 'Antarctic_SAT_1deg_monthly_1979_2023.nc')
    ds1_m.to_netcdf(monthly_path)
    return daily_path, monthly_path


# ============================================================
# Training wrapper (resume fine-tuning when gates fail)
# ============================================================

def run_training(save_dir='./snapshots/antarctica', resume_ckpt=''):
    cmd = ['python', 'train.py', '--save_dir', save_dir, '--log_dir', './logs/antarctica']
    if resume_ckpt:
        cmd += ['--resume', resume_ckpt]
    print('[Train] Launching:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return os.path.join(save_dir, 'ckpt', 'best.pth')


# ============================================================
# Orchestrator
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs_h5', type=str, default='./data/obs/observations_full.h5')

    # Validation gates
    ap.add_argument('--mse_threshold', type=float, default=1.50)
    ap.add_argument('--do_trend_gate', action='store_true')
    ap.add_argument('--trend_min_days', type=int, default=3650)
    ap.add_argument('--trend_min_r', type=float, default=0.35)
    ap.add_argument('--trend_min_sign_agree', type=float, default=0.60)

    # Hold-out
    ap.add_argument('--val_holdout_frac', type=float, default=0.10)
    ap.add_argument('--val_holdout_seed', type=int, default=123)

    # Paths / device
    ap.add_argument('--save_dir', type=str, default='./snapshots/antarctica')
    ap.add_argument('--out_dir',  type=str, default='recon_out')
    ap.add_argument('--device',   type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # 1°×1° options (no xESMF)
    ap.add_argument('--make_1deg', action='store_true',
                    help='Produce 1x1° daily/monthly NetCDF without xESMF')
    ap.add_argument('--ease_latlon', type=str, default='',
                    help='Path to NPZ/NetCDF with 2D EASE lat/lon (keys/vars: lat, lon)')
    ap.add_argument('--one_deg_chunk', type=int, default=64)

    args = ap.parse_args()
    device = torch.device(args.device)

    # Build datasets
    val_ds   = ObsDataset(args.obs_h5, val_holdout_frac=args.val_holdout_frac,
                          val_holdout_seed=args.val_holdout_seed)
    val_ld   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    final_ds = ObsDataset(args.obs_h5, val_holdout_frac=0.0)

    resume = ''
    while True:
        # Train (or resume)
        best_ckpt = run_training(save_dir=args.save_dir, resume_ckpt=resume)

        # Load model
        model = PConvUNet(input_channels=1).to(device)
        load_ckpt(best_ckpt, [('model', model)])
        model.eval()

        # Gate 1
        mse_hold = mse_on_holdout(model, val_ld, device)
        print(f'[Gate-1] hold-out MSE = {mse_hold:.6f}  (≤ {args.mse_threshold})')
        if mse_hold > args.mse_threshold:
            print('[Gate-1] FAIL → resume training …')
            resume = best_ckpt
            continue

        # Gate 2 (optional)
        if args.do_trend_gate:
            print('[Gate-2] computing trends …')
            recon_val, hold_mask = reconstruct_for_validation(model, val_ds, device)
            with h5py.File(args.obs_h5, 'r') as h:
                obs_arr = np.nan_to_num(h['tas'][:], nan=0.0).astype('float32')
            r, frac = trend_gate_on_holdout(recon_val, obs_arr, hold_mask, args.trend_min_days)
            print(f'[Gate-2] r={r:.3f} (≥ {args.trend_min_r}), '
                  f'sign-agree={100*frac:.1f}% (≥ {100*args.trend_min_sign_agree:.1f}%)')
            if (r < args.trend_min_r) or (frac < args.trend_min_sign_agree):
                print('[Gate-2] FAIL → resume training …')
                resume = best_ckpt
                continue

        # Gates passed → final reconstruction & exports
        print('[Final] Reconstructing with ALL observations & exporting …')
        recon_daily = reconstruct_and_save(model, final_ds, device, out_dir=args.out_dir)
        ease_daily_nc = os.path.join(args.out_dir, 'Antarctic_SAT_EASE_daily_1979_2023.nc')
        save_monthlies_ease(recon_daily, out_dir=args.out_dir)

        # Optional 1°×1° products (no xESMF)
        if args.make_1deg:
            if not args.ease_latlon:
                raise ValueError('--make_1deg was set but --ease_latlon was not provided.')
            print('[1deg] Building 1°×1° daily & monthly (no xESMF)…')
            make_1deg_products_without_xesmf(
                ease_daily_nc=ease_daily_nc,
                ease_latlon_path=args.ease_latlon,
                out_dir=args.out_dir,
                chunk=args.one_deg_chunk,
                method='bin'
            )
            print('[1deg] Done.')

        print(f'[Done] See outputs under: ./{args.out_dir}/')
        break


if __name__ == '__main__':
    main()
