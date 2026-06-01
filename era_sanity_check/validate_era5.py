#!/usr/bin/env python
"""Validate Ertel PV (SH gradients, sigma-coordinate) against ERA5 native PV.

Primary: sigma-coordinate PV on 37 ERA5-equivalent σ levels.
Secondary: isobaric PV for reference.

Usage:  micromamba run -n blocking python validate_era5.py
"""
import sys; from pathlib import Path as _Path
_PROJ = _Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path: sys.path.insert(0, str(_PROJ))

import numpy as np, xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.ertel_pv import (ertel_pv_isobaric, ertel_pv_sigma,
                          _interp_to_sigma, DEFAULT_SIGMA_LEVELS)

DATA_DIR = _Path(__file__).resolve().parent / "data"
PLOT_DIR = _Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

era5_f = DATA_DIR / "era5_2025-01-08_00z_pl.nc"
sp_f   = DATA_DIR / "era5_2025-01-08_00z_sp.nc"

# ═══ load ═══
print(f"Loading {era5_f}")
ds = xr.open_dataset(era5_f); ds_sp = xr.open_dataset(sp_f)

u = ds.u.values.squeeze(0); v = ds.v.values.squeeze(0); t = ds.t.values.squeeze(0)
pv_e_si = ds.pv.values.squeeze(0)
lat_orig = ds.latitude.values.astype(float); lon = ds.longitude.values.astype(float)
ph = ds.pressure_level.values.astype(float)     # hPa, descending sfc→top
ps_si = ds_sp.sp.values.squeeze(0)              # Pa

# SH expects lat S→N (ascending), all data surface→top
lat_sh = lat_orig[::-1]
u2 = u[:, ::-1, :][::-1, :, :]
v2 = v[:, ::-1, :][::-1, :, :]
t2 = t[:, ::-1, :][::-1, :, :]
pv_e2 = pv_e_si[:, ::-1, :][::-1, :, :] * 1e6   # → PVU
ps2 = ps_si[::-1, :]
pPa = ph[::-1] * 100.0                            # Pa, ascending top→sfc

print(f"  Shape: {u2.shape}, lat S→N, p top→sfc")
print(f"  PV ERA5 range: [{pv_e2.min():.1f}, {pv_e2.max():.1f}] PVU")

# ═══ isobaric PV (reference) ═══
print("\n── Isobaric PV (SH, reference) ──")
pv_iso = ertel_pv_isobaric(u2, v2, t2, pPa, lat_sh, lon)
print(f"  PV range: [{pv_iso.min():.1f}, {pv_iso.max():.1f}] PVU")

def _find_k(arr, target):
    return np.argmin(np.abs(arr - target))

for hpa in (850, 500, 250):
    k = _find_k(pPa / 100, hpa)
    d = pv_iso[k] - pv_e2[k]
    rmse = np.sqrt(np.nanmean(d**2))
    corr = np.corrcoef(pv_iso[k].ravel(), pv_e2[k].ravel())[0, 1]
    print(f"  {hpa} hPa: RMSE={rmse:.4f} PVU, corr={corr:.4f}")

# ═══ sigma PV (primary) ═══
print("\n── Sigma-coordinate PV (37 σ levels) ──")
sig = DEFAULT_SIGMA_LEVELS  # descending sfc→top
print(f"  σ range: [{sig[-1]:.3f}, {sig[0]:.3f}] ({len(sig)} levels)")

pv_sigma, p_s3d = ertel_pv_sigma(u2, v2, t2, pPa, ps2, lat_sh, lon)
print(f"  PV σ range: [{np.nanmin(pv_sigma):.1f}, {np.nanmax(pv_sigma):.1f}] PVU")

# Interpolate ERA5 native PV to sigma for direct comparison
pv_era5_sig = _interp_to_sigma(pv_e2, pPa, ps2, sig)
p_era5_sig = sig[:, np.newaxis, np.newaxis] * ps2[np.newaxis, :, :]
print(f"  PV ERA5-on-σ range: [{np.nanmin(pv_era5_sig):.1f}, {np.nanmax(pv_era5_sig):.1f}] PVU")
print(f"  NaN fraction in PV σ: {np.isnan(pv_sigma).mean():.4f}")
print(f"  NaN fraction in PV ERA5-on-σ: {np.isnan(pv_era5_sig).mean():.4f}")

for tgt_hpa in (850, 500, 250):
    ks = _find_k(sig, tgt_hpa / 1000)
    ds = pv_sigma[ks] - pv_era5_sig[ks]
    valid = ~np.isnan(ds)
    rmse_s = np.sqrt(np.nanmean(ds[valid]**2)) if valid.any() else np.nan
    pv_s_flat = pv_sigma[ks][valid]; pv_e_flat = pv_era5_sig[ks][valid]
    corr_s = np.corrcoef(pv_s_flat, pv_e_flat)[0, 1] if len(pv_s_flat) > 1 else np.nan
    p_nom = sig[ks] * 101325 / 100  # nominal hPa at std surface
    print(f"  σ={sig[ks]:.3f} (nom {p_nom:.0f} hPa): RMSE={rmse_s:.2f} PVU, corr={corr_s:.4f}")

# ═══ plots ═══
proj = ccrs.Robinson(central_longitude=0); pc = ccrs.PlateCarree()
sig_targets = [0.85, 0.50, 0.25]

fig, axes = plt.subplots(3, 4, figsize=(22, 14), subplot_kw={"projection": proj})
axes = np.atleast_2d(axes)
fig.suptitle("Ertel PV: Sigma-coordinate vs ERA5 (2025-01-08 00Z)",
             fontsize=13, y=0.98)

for row, sig_tgt in enumerate(sig_targets):
    ks = _find_k(sig, sig_tgt)
    hpa_nom = sig[ks] * 101325 / 100
    data_cols = [
        (pv_era5_sig[ks], f"ERA5 PV on σ={sig[ks]:.3f}\n(~{hpa_nom:.0f} hPa nom)"),
        (pv_sigma[ks], f"SH PV on σ={sig[ks]:.3f}\n(~{hpa_nom:.0f} hPa nom)"),
        (pv_sigma[ks] - pv_era5_sig[ks], f"Δ PV (SH−ERA5)\nσ={sig[ks]:.3f}"),
    ]
    # Add isobaric comparison at closest pressure level
    kp = _find_k(pPa / 100, hpa_nom)
    data_cols.append(
        (pv_iso[kp] - pv_e2[kp], f"Δ PV isobaric\n~{pPa[kp]/100:.0f} hPa"),
    )

    for col, (data, title) in enumerate(data_cols):
        ax = axes[row, col]
        is_diff = "Δ" in title
        valid = ~np.isnan(data)
        if valid.sum() > 0:
            vm = np.nanpercentile(np.abs(data[valid]), 99)
        else:
            vm = 1.0
        if is_diff:
            vm = max(vm, 0.1)
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
        cf = ax.pcolormesh(lon, lat_sh, data, cmap="RdBu_r", transform=pc,
                           vmin=-vm, vmax=vm, rasterized=True)
        plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02,
                     label="PVU" if not is_diff else "Δ PVU")
        ax.set_title(title, fontsize=9)

plt.tight_layout()
out = PLOT_DIR / "era5_pv_sigma_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close(fig)

# ═══ RMS profile (sigma) ═══
fig2, ax2 = plt.subplots(figsize=(7, 8))
rms_sigma = np.sqrt(np.nanmean((pv_sigma - pv_era5_sig)**2, axis=(1, 2)))
rms_iso = np.sqrt(np.nanmean((pv_iso - pv_e2)**2, axis=(1, 2)))

ax2.plot(rms_sigma, sig, "r-o", label="Sigma PV vs ERA5-on-σ", markersize=4)
ax2.plot(rms_iso, pPa / 100000, "b-s", label="Isobaric PV vs ERA5", markersize=4)
ax2.set_ylim(1.02, -0.02)
ax2.set_ylabel("σ  [1 = surface]")
ax2.set_xlabel("RMS diff [PVU]")
ax2.set_title("Vertical Profile: PV − ERA5 PV (2025-01-08 00Z)")
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
out2 = PLOT_DIR / "era5_pv_sigma_rms_profile.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("\nDone.")
