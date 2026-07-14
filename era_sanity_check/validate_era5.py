#!/usr/bin/env python
"""Validate Ertel PV against ERA5 native PV — sigma, isentropic, and isobaric.

Three cross-checks against ERA5's own ``pv`` field (2025-01-08 00Z), all with the
**full 3-term** Ertel formula ``-g/p_s·[(f+ζ)∂θ/∂σ + ∂v/∂σ·∂θ/∂x − ∂u/∂σ·∂θ/∂y]``:

1. SIGMA      — PV on 11 terrain-following σ levels vs ERA5 PV interpolated to σ.
2. ISENTROPIC — σ-level PV sampled onto θ = {300,315,320,330,350} K vs ERA5 PV
                interpolated to the same θ (built from ERA5's own T).  [new feature]
3. ISOBARIC   — PV on ERA5 pressure levels vs ERA5 PV, documenting the near-surface
                degradation that the σ approach fixes.

Usage:  micromamba run -n blocking python validate_era5.py
"""
import sys
from pathlib import Path as _Path

_PROJ = _Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.ertel_pv import (
    ertel_pv_sigma, ertel_pv_isobaric, interp_to_isentropic,
    _interp_to_sigma, _compute_theta,
    DEFAULT_SIGMA_LEVELS, DEFAULT_THETA_LEVELS,
)

DATA_DIR = _Path(__file__).resolve().parent / "data"
PLOT_DIR = _Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

era5_f = DATA_DIR / "era5_2025-01-08_00z_pl.nc"
sp_f   = DATA_DIR / "era5_2025-01-08_00z_sp.nc"
PROJ = ccrs.Robinson(central_longitude=0)
PC = ccrs.PlateCarree()


def _corr_rmse(a, b):
    """corr, rmse over common finite points of two 2-D fields."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan, np.nan
    r = np.corrcoef(a[m], b[m])[0, 1]
    rmse = np.sqrt(np.mean((a[m] - b[m]) ** 2))
    return r, rmse


def _map_row(axes, lon, lat, fields, titles, diff_idx=2):
    """Draw one row: [ours | ERA5 | Δ] (or any 3 fields)."""
    for col, (data, title) in enumerate(zip(fields, titles)):
        ax = axes[col]
        is_diff = col == diff_idx
        valid = np.isfinite(data)
        vm = np.nanpercentile(np.abs(data[valid]), 99) if valid.any() else 1.0
        if is_diff:
            vm = max(vm, 0.1)
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
        cf = ax.pcolormesh(lon, lat, data, cmap="RdBu_r", transform=PC,
                           vmin=-vm, vmax=vm, rasterized=True)
        plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02,
                     label="Δ PVU" if is_diff else "PVU")
        ax.set_title(title, fontsize=9)


def _find_k(arr, target):
    return int(np.argmin(np.abs(arr - target)))


# ═══════════════════════════════ load ═══════════════════════════════
print(f"Loading {era5_f}")
ds = xr.open_dataset(era5_f); ds_sp = xr.open_dataset(sp_f)

u = ds.u.values.squeeze(0); v = ds.v.values.squeeze(0); t = ds.t.values.squeeze(0)
pv_e_si = ds.pv.values.squeeze(0)
lat_orig = ds.latitude.values.astype(float); lon = ds.longitude.values.astype(float)
ph = ds.pressure_level.values.astype(float)     # hPa, descending sfc→top
ps_si = ds_sp.sp.values.squeeze(0)              # Pa

# SH expects lat S→N (ascending); all data reordered surface→top? → top→sfc ascending-p
lat_sh = lat_orig[::-1]
u2 = u[:, ::-1, :][::-1, :, :]
v2 = v[:, ::-1, :][::-1, :, :]
t2 = t[:, ::-1, :][::-1, :, :]
pv_e2 = pv_e_si[:, ::-1, :][::-1, :, :] * 1e6   # → PVU
ps2 = ps_si[::-1, :]
pPa = ph[::-1] * 100.0                            # Pa, ascending top→sfc

print(f"  Shape: {u2.shape}, lat S→N, p top→sfc")
print(f"  PV ERA5 range: [{pv_e2.min():.1f}, {pv_e2.max():.1f}] PVU")


# ═══════════════════ 1. SIGMA-coordinate PV (full formula) ═══════════════════
sig = DEFAULT_SIGMA_LEVELS
print(f"\n── (1) Sigma-coordinate PV ({len(sig)} σ levels, method='full') ──")
pv_sigma, p_s3d, theta_s = ertel_pv_sigma(
    u2, v2, t2, pPa, ps2, lat_sh, lon, method="full", return_theta=True)
print(f"  PV σ range: [{np.nanmin(pv_sigma):.1f}, {np.nanmax(pv_sigma):.1f}] PVU")

pv_era5_sig = _interp_to_sigma(pv_e2, pPa, ps2, sig)
print(f"  σ     hPa_nom   RMSE     corr")
for tgt_hpa in (850, 500, 250):
    ks = _find_k(sig, tgt_hpa / 1000)
    r, rmse = _corr_rmse(pv_sigma[ks], pv_era5_sig[ks])
    print(f"  {sig[ks]:.3f}  {sig[ks]*1013:6.0f}   {rmse:5.2f}   {r:.4f}")

# Fig 1: 3×3 sigma maps
fig, axes = plt.subplots(3, 3, figsize=(17, 14), subplot_kw={"projection": PROJ})
fig.suptitle("Ertel PV: Sigma vs ERA5  (full 3-term, 11 σ levels, 2025-01-08 00Z)",
             fontsize=13, y=0.98)
for row, sig_tgt in enumerate([0.85, 0.50, 0.25]):
    ks = _find_k(sig, sig_tgt)
    hpa = sig[ks] * 1013
    _map_row(axes[row], lon, lat_sh,
             [pv_era5_sig[ks], pv_sigma[ks], pv_sigma[ks] - pv_era5_sig[ks]],
             [f"ERA5 PV on σ={sig[ks]:.3f} (~{hpa:.0f} hPa)",
              f"SH PV on σ={sig[ks]:.3f} (~{hpa:.0f} hPa)",
              f"Δ (SH−ERA5)  σ={sig[ks]:.3f}"])
plt.tight_layout()
out = PLOT_DIR / "era5_pv_sigma_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  saved {out}")

# Fig 2: sigma RMS profile
fig2, ax2 = plt.subplots(figsize=(6, 7))
rms_sigma = np.sqrt(np.nanmean((pv_sigma - pv_era5_sig) ** 2, axis=(1, 2)))
ax2.plot(rms_sigma, sig, "r-o", markersize=6)
ax2.set_ylim(1.02, -0.02); ax2.set_ylabel("σ  [1 = surface]")
ax2.set_xlabel("RMS diff [PVU]")
ax2.set_title("Vertical Profile: PV − ERA5 PV  (full formula, 11 σ levels)")
ax2.grid(True, alpha=0.3)
for k in range(len(sig)):
    ax2.annotate(f"{sig[k]:.3f}", (rms_sigma[k], sig[k]),
                 textcoords="offset points", xytext=(8, -2), fontsize=7, alpha=0.7)
plt.tight_layout()
out2 = PLOT_DIR / "era5_pv_sigma_rms_profile.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig2)
print(f"  saved {out2}")


# ═══════════════════ 2. ISENTROPIC PV (σ-PV sampled at θ) ═══════════════════
thL = DEFAULT_THETA_LEVELS
print(f"\n── (2) Isentropic PV on θ={list(thL)} K (σ-PV → θ) ──")
# ours: sample the σ-level PV onto θ using θ on σ levels
pv_theta = interp_to_isentropic(pv_sigma, theta_s, thL)
# ERA5 reference: ERA5 native PV interpolated to θ built from ERA5's own T
theta_era5 = _compute_theta(t2, pPa)             # θ on ERA5 pressure levels
pv_era5_theta = interp_to_isentropic(pv_e2, theta_era5, thL)
print(f"  θ[K]   RMSE    corr    NaN%(ours)")
rms_theta = []
for kk, th in enumerate(thL):
    r, rmse = _corr_rmse(pv_theta[kk], pv_era5_theta[kk])
    rms_theta.append(rmse)
    nanfrac = 100 * np.isnan(pv_theta[kk]).mean()
    print(f"  {th:4.0f}   {rmse:5.2f}   {r:.4f}   {nanfrac:4.1f}")

# Fig 3: 3×3 isentropic maps at θ = 315, 330, 350 K
th_show = [315., 330., 350.]
fig3, axes3 = plt.subplots(3, 3, figsize=(17, 14), subplot_kw={"projection": PROJ})
fig3.suptitle("Ertel PV: Isentropic (σ-PV→θ) vs ERA5-on-θ  (2025-01-08 00Z)",
              fontsize=13, y=0.98)
for row, th in enumerate(th_show):
    kk = _find_k(thL, th)
    _map_row(axes3[row], lon, lat_sh,
             [pv_era5_theta[kk], pv_theta[kk], pv_theta[kk] - pv_era5_theta[kk]],
             [f"ERA5 PV on θ={thL[kk]:.0f} K",
              f"Ours PV on θ={thL[kk]:.0f} K",
              f"Δ (Ours−ERA5)  θ={thL[kk]:.0f} K"])
plt.tight_layout()
out3 = PLOT_DIR / "era5_pv_isentropic_comparison.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight"); plt.close(fig3)
print(f"  saved {out3}")

# Fig 4: isentropic RMS vs θ
fig4, ax4 = plt.subplots(figsize=(6, 6))
ax4.plot(rms_theta, thL, "b-o", markersize=6)
ax4.set_ylabel("θ  [K]"); ax4.set_xlabel("RMS diff [PVU]")
ax4.set_title("Isentropic PV: Ours(σ→θ) − ERA5-on-θ")
ax4.grid(True, alpha=0.3)
for k in range(len(thL)):
    ax4.annotate(f"{thL[k]:.0f} K", (rms_theta[k], thL[k]),
                 textcoords="offset points", xytext=(8, -2), fontsize=8, alpha=0.8)
plt.tight_layout()
out4 = PLOT_DIR / "era5_pv_isentropic_rms.png"
fig4.savefig(out4, dpi=150, bbox_inches="tight"); plt.close(fig4)
print(f"  saved {out4}")


# ═══════════════════ 3. ISOBARIC PV (direct on ERA5 p-levels) ═══════════════════
print(f"\n── (3) Isobaric PV on ERA5 pressure levels (full formula) ──")
pv_iso = ertel_pv_isobaric(u2, v2, t2, pPa, lat_sh, lon, method="full")
print(f"  hPa    RMSE    corr   (isobaric vs ERA5 native pv)")
for tgt_hpa in (850, 500, 250):
    kp = _find_k(pPa, tgt_hpa * 100.0)
    r, rmse = _corr_rmse(pv_iso[kp], pv_e2[kp])
    print(f"  {pPa[kp]/100:4.0f}   {rmse:5.2f}   {r:.4f}")

# Fig 5: 3×3 isobaric maps at 850/500/250 hPa (shows near-surface degradation)
fig5, axes5 = plt.subplots(3, 3, figsize=(17, 14), subplot_kw={"projection": PROJ})
fig5.suptitle("Ertel PV: Isobaric vs ERA5  (full formula, ERA5 p-levels, 2025-01-08 00Z)",
              fontsize=13, y=0.98)
for row, tgt_hpa in enumerate([850, 500, 250]):
    kp = _find_k(pPa, tgt_hpa * 100.0)
    _map_row(axes5[row], lon, lat_sh,
             [pv_e2[kp], pv_iso[kp], pv_iso[kp] - pv_e2[kp]],
             [f"ERA5 PV on {pPa[kp]/100:.0f} hPa",
              f"SH isobaric PV on {pPa[kp]/100:.0f} hPa",
              f"Δ (SH−ERA5)  {pPa[kp]/100:.0f} hPa"])
plt.tight_layout()
out5 = PLOT_DIR / "era5_pv_isobaric_comparison.png"
fig5.savefig(out5, dpi=150, bbox_inches="tight"); plt.close(fig5)
print(f"  saved {out5}")

print("\nDone.")
