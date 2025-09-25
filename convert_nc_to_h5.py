
"""
Antarctic SAT prep: NetCDF → HDF5 (splits + full obs)

What this script does
- Loads three NetCDFs in ./data:
    • normalized_reanalysis_data.nc     (expects var 'normalized_air' or first var)
    • obs_mask_ease_100km_180x180.nc    (expects var 'obs_mask' or first var)
    • normalized_observation_data.nc    (expects var 'normalized_SAT' or first var)
- Normalizes dims to [time, row, col] (handles 'date'/'month' quirks).
- Writes reanalysis + masks into train/val/test HDF5 files.
- Writes the full (UNSPLIT) observations into a single HDF5.

Outputs (dataset name: 'tas', dtype float32)
- ./data/train_large/reanalysis_train.h5
- ./data/val_large/reanalysis_val.h5
- ./data/test_large/reanalysis_test.h5
- ./data/mask/mask_{train,val,test}.h5
- ./data/obs/observations_full.h5   (entire 1979–2023 observation cube, unsplit)

Splits (inclusive start, inclusive end)
- train: 1979-01-01 → 2005-12-31
- val  : 2006-01-01 → 2012-12-31
- test : 2013-01-01 → 2023-12-31

Notes
- Mask is time-aligned to reanalysis per split using nearest day (≤1D tolerance), then NaNs→0.
- If expected variable names aren’t present, the first data var is used.
- Values are assumed to be already normalized (z-scores).
- HDF5 datasets include labeled dims: time / lat / lon (labels are optional metadata).

Usage
- Put the 3 NetCDF files under ./data, then run:
    python nc_to_h5.py
- Safe to re-run; files will be overwritten.

Dependencies
- xarray, h5py, numpy
"""
import os
import h5py
import numpy as np
import xarray as xr

# ------------ Inputs (all from ./data) ------------
DATA_DIR = "./data"
REAN_NC  = os.path.join(DATA_DIR, "normalized_reanalysis_data.nc")
MASK_NC  = os.path.join(DATA_DIR, "obs_mask_ease_100km_180x180.nc")
OBS_NC   = os.path.join(DATA_DIR, "normalized_observation_data.nc")  # <- NEW (unsplit)

# ------------ Splits (for reanalysis + mask only) ------------
SPLITS = {
    'train': ('1979-01-01', '2005-12-31'),
    'val'  : ('2006-01-01', '2012-12-31'),
    'test' : ('2013-01-01', '2023-12-31'),
}

# ------------ Outputs ------------
OUTS = {
    'train': (os.path.join(DATA_DIR, "train_large/reanalysis_train.h5"),
              os.path.join(DATA_DIR, "mask/mask_train.h5")),
    'val'  : (os.path.join(DATA_DIR, "val_large/reanalysis_val.h5"),
              os.path.join(DATA_DIR, "mask/mask_val.h5")),
    'test' : (os.path.join(DATA_DIR, "test_large/reanalysis_test.h5"),
              os.path.join(DATA_DIR, "mask/mask_test.h5")),
}

# Single, unsplit observations output
OBS_OUT_H5 = os.path.join(DATA_DIR, "obs/observations_full.h5")

# ------------ Small utilities ------------
def write_h5(arr3d: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, 'w') as h:
        d = h.create_dataset('tas', data=arr3d.astype('float32'), dtype='f4')
        # optional labels (not all loaders read these, but harmless)
        d.dims[0].label = 'time'
        d.dims[1].label = 'lat'
        d.dims[2].label = 'lon'

def _rename_time_if_needed(da: xr.DataArray) -> xr.DataArray:
    
    if 'time' in da.coords:
        return da
    if 'date' in da.coords:
        return da.rename({'date': 'time'})
    # if neither exists but a datetime coord is present as a dim name
    for d in da.dims:
        if np.issubdtype(da[d].dtype, np.datetime64):
            return da.rename({d: 'time'})
    return da  # fall back (later logic will try to detect)

def ensure_time_row_col(da: xr.DataArray) -> xr.DataArray:
    
    da = da.squeeze(drop=True)
    da = _rename_time_if_needed(da)

    # Drop any month-like dimension by aligning with time.month if possible
    month_like = [d for d in da.dims if d.lower() in ('month', 'mon', 'mn')]
    if month_like:
        d = month_like[0]
        if 'time' in da.coords:
            # Select the correct month per timestamp, then drop the month axis
            da = da.transpose('time', *[x for x in da.dims if x != 'time'])
            da = da.sel({d: da['time'].dt.month}).drop(d)
        else:
            # No time to align with — take the first slice
            da = da.isel({d: 0}).drop(d)

    # Work out time dimension again (some datasets use 'date')
    time_dim = None
    for d in da.dims:
        if d == 'time' or np.issubdtype(da[d].dtype, np.datetime64):
            time_dim = d
            break
    if time_dim is None:
        # Fallback: treat longest axis as time
        time_dim = max(da.sizes, key=lambda k: da.sizes[k])

    # Figure out spatial dims
    spatial_candidates = ['grid_row', 'grid_col', 'row', 'col', 'y', 'x', 'lat', 'lon']
    rest = [d for d in da.dims if d != time_dim]
    if len(rest) != 2:
        # Try to guess from common spatial names
        found = [d for d in da.dims if d in spatial_candidates]
        if len(found) == 2:
            rest = found
        else:
            raise ValueError(f'Expected 3 dims after cleanup, got {da.dims}')

    row_dim, col_dim = rest
    return da.transpose(time_dim, row_dim, col_dim)

def main():
    # ---------- Load reanalysis ----------
    ds_r = xr.open_dataset(REAN_NC)
    var_r = 'normalized_air' if 'normalized_air' in ds_r.data_vars else list(ds_r.data_vars)[0]
    da_r = ensure_time_row_col(ds_r[var_r])

    # ---------- Load mask ----------
    ds_m = xr.open_dataset(MASK_NC)
    var_m = 'obs_mask' if 'obs_mask' in ds_m.data_vars else list(ds_m.data_vars)[0]
    da_m = ensure_time_row_col(ds_m[var_m])

    # ---------- Load observations (UNSPLIT) ----------
    ds_o = xr.open_dataset(OBS_NC)
    var_o = 'normalized_SAT' if 'normalized_SAT' in ds_o.data_vars else list(ds_o.data_vars)[0]
    da_o = ensure_time_row_col(ds_o[var_o])

    # Save the full observations to a single H5 (no split)
    write_h5(da_o.astype('float32').values, OBS_OUT_H5)
    print(f"observations_full: {tuple(da_o.shape)} → {OBS_OUT_H5}")

    # ---------- Per-split Reanalysis + Mask ----------
    for split, (t0, t1) in SPLITS.items():
        rean = da_r.sel(time=slice(t0, t1)).astype('float32')

        # Align mask to reanalysis time (nearest day, 1-day tolerance), fill missing with 0
        mask = (
            da_m.reindex(time=rean.time, method='nearest', tolerance='1D')
                .fillna(0.0)
                .astype('float32')
        )

        out_rean, out_mask = OUTS[split]
        write_h5(rean.values, out_rean)
        write_h5(mask.values, out_mask)

        print(f"{split}: rean {tuple(rean.shape)}  mask {tuple(mask.shape)}  →  {OUTS[split]}")

if __name__ == '__main__':
    main()
