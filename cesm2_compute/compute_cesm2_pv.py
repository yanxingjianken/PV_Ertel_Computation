#!/usr/bin/env python
"""Download CESM2-LENS2 daily U, V, T, PS from AWS S3 and compute Ertel PV.

Sigma-coordinate PV on 11 essential σ levels (σ = p / p_s).
Isobaric PV is NOT computed — sigma is the sole coordinate.

Usage
-----
    micromamba run -n blocking python compute_cesm2_pv.py
"""

import sys
from pathlib import Path as _Path

_PROJ = _Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np
import xarray as xr
import s3fs
import cftime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.ertel_pv import ertel_pv_sigma, DEFAULT_SIGMA_LEVELS

# ---- config ----
DATA_DIR = _Path(__file__).resolve().parent / "data"
PLOT_DIR = _Path(__file__).resolve().parent / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

S3_PREFIX = "s3://ncar-cesm2-lens/atm/daily"
EXPERIMENT = "cesm2LE-historical-cmip6"
OUT_NC = DATA_DIR / "cesm2le_hist_daily_uvtps_sample.nc"

# ---- download ----
print("Connecting to AWS S3 (anonymous) ...")
fs = s3fs.S3FileSystem(anon=True)

print("Opening U store ...")
ds_u = xr.open_zarr(fs.get_mapper(f"{S3_PREFIX}/{EXPERIMENT}-U.zarr"), consolidated=True)
print("Opening V store ...")
ds_v = xr.open_zarr(fs.get_mapper(f"{S3_PREFIX}/{EXPERIMENT}-V.zarr"), consolidated=True)
print("Opening T store ...")
ds_t = xr.open_zarr(fs.get_mapper(f"{S3_PREFIX}/{EXPERIMENT}-T.zarr"), consolidated=True)
print("Opening PS store ...")
ds_ps = xr.open_zarr(fs.get_mapper(f"{S3_PREFIX}/{EXPERIMENT}-PS.zarr"), consolidated=True)

# Align member_ids across all stores
common_members = np.intersect1d(
    np.intersect1d(
        np.intersect1d(ds_u.member_id.values, ds_v.member_id.values),
        ds_t.member_id.values,
    ),
    ds_ps.member_id.values,
)
print(f"Common members: {len(common_members)}")
member = common_members[0]
member_idx_u = np.where(ds_u.member_id.values == member)[0][0]
member_idx_v = np.where(ds_v.member_id.values == member)[0][0]
member_idx_t = np.where(ds_t.member_id.values == member)[0][0]
member_idx_ps = np.where(ds_ps.member_id.values == member)[0][0]
print(f"Selected member: {member}")
print(f"U shape: {ds_u.U.shape}")
print(f"PS shape: {ds_ps.PS.shape}")
print(f"Levels (hPa): {ds_u.lev.values.astype(float)[:5]} ... "
      f"{ds_u.lev.values.astype(float)[-5:]}")
print(f"Time range: {ds_u.time.values[0]} ... {ds_u.time.values[-1]}")

# Select 2010-01-01
target_date = cftime.DatetimeNoLeap(2010, 1, 1)
time_vals = ds_u.time.values
time_ordinals = np.array([
    t.toordinal() if hasattr(t, 'toordinal')
    else cftime.date2num(t, 'days since 1850-01-01')
    for t in time_vals
])
target_ordinal = target_date.toordinal() if hasattr(target_date, 'toordinal') else 0
time_idx = np.argmin(np.abs(time_ordinals - target_ordinal))
actual_date = time_vals[time_idx]
print(f"Selected time: {actual_date} (idx={time_idx})")

# Subset
print("Subsetting data ...")
u_sub = ds_u.U.isel(member_id=member_idx_u, time=time_idx)
v_sub = ds_v.V.isel(member_id=member_idx_v, time=time_idx)
t_sub = ds_t.T.isel(member_id=member_idx_t, time=time_idx)
ps_sub = ds_ps.PS.isel(member_id=member_idx_ps, time=time_idx)

print(f"  u shape: {u_sub.shape}")
print(f"  ps shape: {ps_sub.shape}")
print(f"  ps range: [{float(ps_sub.min()):.0f}, {float(ps_sub.max()):.0f}] Pa")

# ---- save ----
print(f"\nSaving to {OUT_NC} ...")
ds_out = xr.Dataset({
    "u": u_sub, "v": v_sub, "t": t_sub, "ps": ps_sub,
})
ds_out.to_netcdf(OUT_NC)
print(f"  Saved: {OUT_NC} ({OUT_NC.stat().st_size / 1e6:.1f} MB)")

# ---- compute PV ----
print(f"\nComputing Ertel PV (sigma-coordinate, {len(DEFAULT_SIGMA_LEVELS)} levels) ...")
u_arr = u_sub.values.astype(float)
v_arr = v_sub.values.astype(float)
t_arr = t_sub.values.astype(float)
ps_arr = ps_sub.values.astype(float)
lat = u_sub.lat.values.astype(float)
lon = u_sub.lon.values.astype(float)

# CESM2 lev: top→surface, ascending index. Flip to surface→top.
lev_hPa = u_sub.lev.values.astype(float)
u_arr = u_arr[::-1, :, :]; v_arr = v_arr[::-1, :, :]; t_arr = t_arr[::-1, :, :]
plev_Pa = lev_hPa[::-1] * 100.0
print(f"  plev: {plev_Pa[0]:.0f} → {plev_Pa[-1]:.0f} Pa  (sfc→top)")

# Sigma PV (sole product)
pv_sigma, p_s3d = ertel_pv_sigma(u_arr, v_arr, t_arr, plev_Pa, ps_arr, lat, lon)
print(f"  PV σ range: [{np.nanmin(pv_sigma):.1f}, {np.nanmax(pv_sigma):.1f}] PVU")

# ---- plots ----
proj = ccrs.Robinson(central_longitude=0); pc = ccrs.PlateCarree()
sig = DEFAULT_SIGMA_LEVELS
date_str = f"{actual_date.year:04d}-{actual_date.month:02d}-{actual_date.day:02d}"

# Sigma PV at 3 levels
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
for ax, sig_tgt in zip(axes, [0.85, 0.50, 0.25]):
    ks = np.argmin(np.abs(sig - sig_tgt))
    data = pv_sigma[ks]
    vm = np.nanpercentile(np.abs(data), 99)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
    cf = ax.pcolormesh(lon, lat, data, cmap="RdBu_r", transform=pc,
                       vmin=-vm, vmax=vm, rasterized=True)
    plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02, label="PVU")
    ax.set_title(f"CESM2-LE PV  σ={sig[ks]:.3f}  (nom {sig[ks]*1013:.0f} hPa)\n"
                 f"({member}, {date_str})", fontsize=9)

fig.suptitle(f"COSM2-LENS2 Ertel PV (sigma-coordinate, {len(sig)} levels)",
             fontsize=13, y=0.98)
plt.tight_layout()
out_png = PLOT_DIR / "cesm2le_pv_sigma.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_png}")
plt.close(fig)

print(f"\nDone. CESM2-LENS2 PV computation complete.")
print(f"  Member: {member}, Date: {date_str}")
print(f"  Data:  {OUT_NC}")
print(f"  Plots: {PLOT_DIR}")
