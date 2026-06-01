#!/usr/bin/env python
"""Download CESM2-LENS2 daily U, V, T data from AWS S3 and compute Ertel PV.

Downloads one timestep (one member, one date) for testing, saves to local
NetCDF, then computes Ertel PV using ``ertel_pv_isobaric()``.

Usage
-----
    micromamba run -n mpas_toolchain python compute_cesm2_pv.py
"""

import sys
from pathlib import Path as _Path

_PROJ = _Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np
import xarray as xr
import s3fs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.ertel_pv import ertel_pv_isobaric

# ---- config ----
DATA_DIR = _Path(__file__).resolve().parent / "data"
PLOT_DIR = _Path(__file__).resolve().parent / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# CESM2-LENS2 AWS S3 paths (daily, historical, CMIP6)
S3_PREFIX = "s3://ncar-cesm2-lens/atm/daily"
EXPERIMENT = "cesm2LE-historical-cmip6"

# Pick one member and one date
MEMBER_IDX = 0       # first member (r10i1181p1f1)
TIME_IDX = -365       # last year; use -365 for ~1 year before end (2013-12-31→2014-12-31)
# Actually let's use a specific index: 2010-01-01 is roughly day 58440 from 1850-01-01
# We'll search for it instead

OUT_NC = DATA_DIR / "cesm2le_hist_daily_uvt_sample.nc"

# ---- download ----
print("Connecting to AWS S3 (anonymous) ...")
fs = s3fs.S3FileSystem(anon=True)

# Open U, V, T stores
print("Opening U store ...")
ds_u = xr.open_zarr(fs.get_mapper(f"{S3_PREFIX}/{EXPERIMENT}-U.zarr"), consolidated=True)
print("Opening V store ...")
ds_v = xr.open_zarr(fs.get_mapper(f"{S3_PREFIX}/{EXPERIMENT}-V.zarr"), consolidated=True)
print("Opening T store ...")
ds_t = xr.open_zarr(fs.get_mapper(f"{S3_PREFIX}/{EXPERIMENT}-T.zarr"), consolidated=True)

# Align member_ids between U, V, T (U has 49, V/T have 50 members)
# Use the intersection of member_ids
common_members = np.intersect1d(
    np.intersect1d(ds_u.member_id.values, ds_v.member_id.values),
    ds_t.member_id.values
)
print(f"Common members: {len(common_members)}")
member = common_members[0]
member_idx_u = np.where(ds_u.member_id.values == member)[0][0]
member_idx_v = np.where(ds_v.member_id.values == member)[0][0]
member_idx_t = np.where(ds_t.member_id.values == member)[0][0]
print(f"\nSelected member: {member} (u_idx={member_idx_u}, v_idx={member_idx_v}, t_idx={member_idx_t})")
print(f"U shape: {ds_u.U.shape} (member×time×lev×lat×lon)")
print(f"V shape: {ds_v.V.shape}")
print(f"T shape: {ds_t.T.shape}")
print(f"Levels (hPa): {ds_u.lev.values[:5]} ... {ds_u.lev.values[-5:]}")
print(f"Lat: {ds_u.lat.values[:3]} ... {ds_u.lat.values[-3:]}")
print(f"Time range: {ds_u.time.values[0]} ... {ds_u.time.values[-1]}")

# Select one timestep (search for 2010-01-01)
# CESM uses no-leap calendar; use cftime for comparison
import cftime
target_date = cftime.DatetimeNoLeap(2010, 1, 1)
time_vals = ds_u.time.values  # cftime array
# Convert to ordinal days for argmin
import numpy as np
time_ordinals = np.array([t.toordinal() if hasattr(t, 'toordinal') else
                          cftime.date2num(t, 'days since 1850-01-01')
                          for t in time_vals])
target_ordinal = target_date.toordinal() if hasattr(target_date, 'toordinal') else 0
time_idx = np.argmin(np.abs(time_ordinals - target_ordinal))
actual_date = time_vals[time_idx]
print(f"\nSelected time: {actual_date} (idx={time_idx})")

# Subset
print("Subsetting data (1 member, 1 time, all levels, full lat-lon) ...")
u_sub = ds_u.U.isel(member_id=member_idx_u, time=time_idx)
v_sub = ds_v.V.isel(member_id=member_idx_v, time=time_idx)
t_sub = ds_t.T.isel(member_id=member_idx_t, time=time_idx)

print(f"  u shape: {u_sub.shape}")
print(f"  u range: [{float(u_sub.min()):.1f}, {float(u_sub.max()):.1f}] m/s")
print(f"  t range: [{float(t_sub.min()):.1f}, {float(t_sub.max()):.1f}] K")

# Check level ordering — CESM2 lev is top→surface (ascending index = increasing pressure)
# Our code expects surface→top (decreasing pressure along index)
lev_hPa = u_sub.lev.values.astype(float)  # hPa, top→surface
print(f"\nLevel direction: {'top→surface' if lev_hPa[0] < lev_hPa[-1] else 'surface→top'}")

# ---- save to NetCDF ----
print(f"\nSaving to {OUT_NC} ...")
ds_out = xr.Dataset({
    "u": u_sub,
    "v": v_sub,
    "t": t_sub,
})
ds_out.to_netcdf(OUT_NC)
print(f"  Saved: {OUT_NC} ({OUT_NC.stat().st_size / 1e6:.1f} MB)")

# ---- compute PV ----
print("\nComputing Ertel PV ...")
u_arr = u_sub.values      # (32, 192, 288)
v_arr = v_sub.values
t_arr = t_sub.values
lat = u_sub.lat.values    # degrees
lon = u_sub.lon.values

# Flip from top→surface to surface→top for our code
# CESM2 lev: 3.6 → 992.6 hPa (top→surface, but stored as ascending index)
# After flip: 992.6 → 3.6 hPa (surface→top)
u_arr = u_arr[::-1, :, :]
v_arr = v_arr[::-1, :, :]
t_arr = t_arr[::-1, :, :]
plev_Pa = lev_hPa[::-1] * 100.0  # hPa → Pa, surface→top

print(f"  plev after flip: {plev_Pa[0]:.0f} → {plev_Pa[-1]:.0f} Pa")
print(f"  Data shape: {u_arr.shape}")

# Compute PV (simple formula for best match with standard pressure-level products)
pv_full = ertel_pv_isobaric(u_arr, v_arr, t_arr, plev_Pa, lat, lon, method="full")
pv_simple = ertel_pv_isobaric(u_arr, v_arr, t_arr, plev_Pa, lat, lon, method="simple")

print(f"\n  PV full   range: [{pv_full.min():.2f}, {pv_full.max():.2f}] PVU")
print(f"  PV simple range: [{pv_simple.min():.2f}, {pv_simple.max():.2f}] PVU")

# ---- quick-look plots ----
print("\nGenerating quick-look plots ...")
proj = ccrs.Robinson(central_longitude=0)
pc = ccrs.PlateCarree()

# Find level indices for approximate 850, 500, 250 hPa
def find_lev_idx(plev_Pa, target_hPa):
    return np.argmin(np.abs(plev_Pa / 100 - target_hPa))

lev_targets = [850, 500, 250]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})

for ax, target_hPa in zip(axes, lev_targets):
    k = find_lev_idx(plev_Pa, target_hPa)
    actual_hPa = plev_Pa[k] / 100
    pv_level = pv_simple[k]
    
    vm = np.nanpercentile(np.abs(pv_level), 99)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
    cf = ax.pcolormesh(lon, lat, pv_level, cmap="RdBu_r", transform=pc,
                       vmin=-vm, vmax=vm, rasterized=True)
    plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02, label="PVU")
    date_str = f"{actual_date.year:04d}-{actual_date.month:02d}-{actual_date.day:02d}"
    ax.set_title(f"CESM2-LE PV @ ~{actual_hPa:.0f} hPa\n"
                 f"({member}, {date_str})", fontsize=9)

fig.suptitle("CESM2-LENS2 Ertel PV (Simple Formula)", fontsize=13, y=0.98)
plt.tight_layout()
out_png = PLOT_DIR / "cesm2le_pv_quicklook.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"  Saved: {out_png}")
plt.close(fig)

print(f"\n✓ CESM2-LENS2 PV computation complete.")
date_str = f"{actual_date.year:04d}-{actual_date.month:02d}-{actual_date.day:02d}"
print(f"  Member: {member}, Date: {date_str}")
print(f"  Data:  {OUT_NC}")
print(f"  Plots: {PLOT_DIR}")
