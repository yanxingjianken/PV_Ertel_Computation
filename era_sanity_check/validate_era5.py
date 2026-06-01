#!/usr/bin/env python
"""Validate MPAS-like Ertel PV computation against ERA5 native PV.

Computes Ertel PV from ERA5 u, v, T on pressure levels using both the
full 3-term formula and the simplified (f+ζ)∂θ/∂p formula, then compares
against ERA5's native ``pv`` field at lower (850 hPa), mid (500 hPa),
and upper (250 hPa) levels.

Usage
-----
    micromamba run -n mpas_toolchain python validate_era5.py
"""

import sys
from pathlib import Path as _Path

# Add project root to path
_PROJ = _Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.ertel_pv import ertel_pv_isobaric

# ---- paths ----
DATA_DIR = _Path(__file__).resolve().parent / "data"
PLOT_DIR = _Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ERA5_FILE = DATA_DIR / "era5_2025-01-08_00z_pl.nc"

# ---- load data ----
print(f"Loading: {ERA5_FILE}")
ds = xr.open_dataset(ERA5_FILE)

# ERA5 CDS API returns pressure_level in hPa, lat N→S (90→-90)
u = ds["u"].values  # (time, lev, lat, lon)
v = ds["v"].values
t = ds["t"].values
pv_era5_si = ds["pv"].values  # ERA5 native PV in SI (K m² kg⁻¹ s⁻¹)

plev_hpa = ds["pressure_level"].values.astype(float)  # hPa
lat = ds["latitude"].values.astype(float)             # °N
lon = ds["longitude"].values.astype(float)            # °E

# Squeeze time dim (single timestep)
u = u.squeeze(axis=0)   # (lev, lat, lon)
v = v.squeeze(axis=0)
t = t.squeeze(axis=0)
pv_era5_si = pv_era5_si.squeeze(axis=0)
# Convert ERA5 PV from SI → PVU (1 PVU = 10⁻⁶ K m² kg⁻¹ s⁻¹)
pv_era5 = pv_era5_si * 1.0e6

# Convert pressure to Pa (ascending: surface → top → need to flip since
# CDS returns descending hPa: 1000 → 1)
plev_Pa = plev_hpa[::-1] * 100.0  # now 1→1000 hPa in Pa

# Flip data to match ascending pressure (surface at index 0)
# CDS returns 1000→1 hPa (surface→top), which IS surface-first.
# Wait — CDS returns lat 90→-90 (N→S) and pressure_level sorted descending?
# Let's check and handle.
# Actually CDS pressure_level order is what we requested. We requested
# 1,2,3,...,1000 → so plev_hpa is 1→1000 (top→surface).
# Our code expects surface→top (high p → low p), so we need to flip.
print(f"Pressure levels (original): {plev_hpa[:5]} ... {plev_hpa[-5:]} hPa")

# Flip: surface → top (1000→1 hPa)
flip_lev = True
if plev_hpa[0] < plev_hpa[-1]:
    # Currently top→surface (1→1000), flip to surface→top
    u = u[::-1, :, :]
    v = v[::-1, :, :]
    t = t[::-1, :, :]
    pv_era5 = pv_era5[::-1, :, :]
    plev_Pa = plev_hpa[::-1] * 100.0
    print(f"Flipped levels: {plev_Pa[0]:.0f} → {plev_Pa[-1]:.0f} Pa")
else:
    plev_Pa = plev_hpa * 100.0
    flip_lev = False
    print(f"Levels already surface→top: {plev_Pa[0]:.0f} → {plev_Pa[-1]:.0f} Pa")

print(f"Data shape: {u.shape} (lev={u.shape[0]}, lat={u.shape[1]}, lon={u.shape[2]})")
print(f"Lat range: {lat[0]:.1f} to {lat[-1]:.1f}")
print(f"u range: [{u.min():.1f}, {u.max():.1f}] m/s")
print(f"T range: [{t.min():.1f}, {t.max():.1f}] K")
print(f"ERA5 PV range: [{pv_era5.min():.2f}, {pv_era5.max():.2f}] PVU")

# ---- compute PV ----
print("\nComputing Ertel PV (full 3-term formula) ...")
pv_full = ertel_pv_isobaric(u, v, t, plev_Pa, lat, lon, method="full")

print("Computing Ertel PV (simple (f+ζ)∂θ/∂p only) ...")
pv_simple = ertel_pv_isobaric(u, v, t, plev_Pa, lat, lon, method="simple")

print(f"\nPV full   range: [{pv_full.min():.2f}, {pv_full.max():.2f}] PVU")
print(f"PV simple range: [{pv_simple.min():.2f}, {pv_simple.max():.2f}] PVU")
print(f"ERA5 PV   range: [{pv_era5.min():.2f}, {pv_era5.max():.2f}] PVU")

# ---- compare at key levels ----
# Find level indices for 850, 500, 250 hPa
def find_lev(plev_Pa, target_hPa):
    """Find index of pressure level closest to target hPa."""
    target_Pa = target_hPa * 100.0
    return np.argmin(np.abs(plev_Pa - target_Pa))

levs = {
    "850 hPa": find_lev(plev_Pa, 850),
    "500 hPa": find_lev(plev_Pa, 500),
    "250 hPa": find_lev(plev_Pa, 250),
}

for name, k in levs.items():
    p_actual = plev_Pa[k] / 100
    diff_full = pv_full[k] - pv_era5[k]
    diff_simple = pv_simple[k] - pv_era5[k]
    print(f"\n--- {name} (level {k}, p={p_actual:.0f} hPa) ---")
    print(f"  PV full   vs ERA5: RMSE={np.sqrt(np.nanmean(diff_full**2)):.4f} PVU, "
          f"bias={np.nanmean(diff_full):.4f}, corr={np.corrcoef(pv_full[k].ravel(), pv_era5[k].ravel())[0,1]:.4f}")
    print(f"  PV simple vs ERA5: RMSE={np.sqrt(np.nanmean(diff_simple**2)):.4f} PVU, "
          f"bias={np.nanmean(diff_simple):.4f}, corr={np.corrcoef(pv_simple[k].ravel(), pv_era5[k].ravel())[0,1]:.4f}")

# ---- plot ----
print("\nGenerating comparison plots ...")

proj = ccrs.Robinson(central_longitude=0)
pc = ccrs.PlateCarree()

fig, axes = plt.subplots(3, 4, figsize=(22, 14),
                          subplot_kw={"projection": proj})
axes = np.atleast_2d(axes)

for row, (name, k) in enumerate(levs.items()):
    p_actual = plev_Pa[k] / 100
    data_list = [
        (pv_era5[k], f"ERA5 native PV\n{name} ({p_actual:.0f} hPa)"),
        (pv_full[k], f"Computed PV (full)\n{name} ({p_actual:.0f} hPa)"),
        (pv_simple[k], f"Computed PV (simple)\n{name} ({p_actual:.0f} hPa)"),
        (pv_full[k] - pv_era5[k], f"Diff (full − ERA5)\n{name} ({p_actual:.0f} hPa)"),
    ]

    for col, (data, title) in enumerate(data_list):
        ax = axes[row, col]
        is_diff = "Diff" in title
        vm = np.nanpercentile(np.abs(data), 99) if not is_diff else np.nanpercentile(np.abs(data), 99)
        if is_diff:
            vm = max(vm, 0.1)  # avoid zero range
            cmap = "RdBu_r"
        else:
            cmap = "RdBu_r"

        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
        cf = ax.pcolormesh(lon, lat, data, cmap=cmap, transform=pc,
                           vmin=-vm, vmax=vm, rasterized=True)
        plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02,
                     label="PVU" if not is_diff else "Δ PVU")
        ax.set_title(title, fontsize=9)

fig.suptitle("Ertel PV Validation: Computed vs ERA5 Native (2025-01-08 00Z)",
             fontsize=13, y=0.98)
plt.tight_layout()
out_png = PLOT_DIR / "era5_pv_comparison.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"  Saved: {out_png}")
plt.close(fig)

# ---- vertical profile of RMS difference ----
rms_full = np.sqrt(np.nanmean((pv_full - pv_era5)**2, axis=(1, 2)))
rms_simple = np.sqrt(np.nanmean((pv_simple - pv_era5)**2, axis=(1, 2)))

fig2, ax2 = plt.subplots(figsize=(7, 8))
ax2.plot(rms_full, plev_Pa / 100, "b-o", label="Full formula", markersize=4)
ax2.plot(rms_simple, plev_Pa / 100, "r--s", label="Simple (f+ζ)∂θ/∂p", markersize=4)
ax2.set_ylim(plev_Pa[-1] / 100, plev_Pa[0] / 100)
ax2.set_ylabel("Pressure [hPa]")
ax2.set_xlabel("RMS difference [PVU]")
ax2.set_title("Vertical Profile: Computed PV − ERA5 Native PV\n(2025-01-08 00Z)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
out_png2 = PLOT_DIR / "era5_pv_rms_profile.png"
fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
print(f"  Saved: {out_png2}")
plt.close(fig2)

print("\n✓ Validation complete.")
