#!/usr/bin/env python
"""Validate MPAS-like Ertel PV (SH gradients) against ERA5 native PV.

Compares isobaric PV computed with spherical-harmonic horizontal
derivatives (pvtend.sh_ops) against ERA5 native PV at 850, 500, 250 hPa.

Usage:  micromamba run -n blocking python validate_era5.py
"""
import sys; from pathlib import Path as _Path
_PROJ = _Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path: sys.path.insert(0, str(_PROJ))

import numpy as np, xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from src.ertel_pv import ertel_pv_isobaric, ertel_pv_sigma, _interp_to_sigma

DATA_DIR = _Path(__file__).resolve().parent / "data"
PLOT_DIR = _Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

era5_f = DATA_DIR / "era5_2025-01-08_00z_pl.nc"
sp_f   = DATA_DIR / "era5_2025-01-08_00z_sp.nc"

# ---- load ----
print(f"Loading {era5_f}")
ds = xr.open_dataset(era5_f)
ds_sp = xr.open_dataset(sp_f)

u=ds.u.values.squeeze(0); v=ds.v.values.squeeze(0); t=ds.t.values.squeeze(0)
pv_e_si=ds.pv.values.squeeze(0)
lat_orig=ds.latitude.values.astype(float); lon=ds.longitude.values.astype(float)
ph=ds.pressure_level.values.astype(float)
ps_si=ds_sp.sp.values.squeeze(0)

# SH expects S→N (ascending), surface→top
lat_sh = lat_orig[::-1]
u2=u[:,::-1,:][::-1,:,:]; v2=v[:,::-1,:][::-1,:,:]; t2=t[:,::-1,:][::-1,:,:]
pv_e2=pv_e_si[:,::-1,:][::-1,:,:]*1e6; ps2=ps_si[::-1,:]; pPa=ph[::-1]*100.0

print(f"Shape: {u2.shape}, lat S→N, p surface→top")
print(f"PV ERA5 range: [{pv_e2.min():.1f}, {pv_e2.max():.1f}] PVU")

# ---- isobaric PV (SH) ----
print("\nComputing SH isobaric PV ...")
pv_sh = ertel_pv_isobaric(u2,v2,t2,pPa,lat_sh,lon)
print(f"PV SH range: [{pv_sh.min():.1f}, {pv_sh.max():.1f}] PVU")

# ---- compare ----
def find_k(pPa, hpa):
    return np.argmin(np.abs(pPa/100 - hpa))

for name, hpa in [("850 hPa", 850), ("500 hPa", 500), ("250 hPa", 250)]:
    k = find_k(pPa, hpa)
    d = pv_sh[k] - pv_e2[k]
    rmse = np.sqrt(np.nanmean(d**2))
    bias = np.nanmean(d)
    corr = np.corrcoef(pv_sh[k].ravel(), pv_e2[k].ravel())[0,1]
    print(f"  {name}: RMSE={rmse:.4f} PVU, bias={bias:+.4f}, corr={corr:.4f}")

# ---- sigma PV (experimental) ----
print("\nComputing sigma PV (experimental) ...")
try:
    pv_sig, p_s3d = ertel_pv_sigma(u2,v2,t2,pPa,ps2,lat_sh,lon)
    pv_era5_sig = _interp_to_sigma(pv_e2,pPa,ps2,pPa/100000.0)
    print(f"PV sig range: [{pv_sig.min():.1f}, {pv_sig.max():.1f}] PVU")
    
    sig_levels = pPa / 100000.0
    for name, hpa in [("850 hPa", 850), ("500 hPa", 500), ("250 hPa", 250)]:
        ks = np.argmin(np.abs(sig_levels - hpa/1000))
        d_s = pv_sig[ks] - pv_era5_sig[ks]
        rmse_s = np.sqrt(np.nanmean(d_s**2))
        corr_s = np.corrcoef(pv_sig[ks].ravel(), pv_era5_sig[ks].ravel())[0,1]
        print(f"  σ-{name}: RMSE={rmse_s:.4f} PVU, corr={corr_s:.4f}")
except Exception as e:
    print(f"  Sigma PV failed: {e}")
    pv_sig = None

# ---- plots ----
proj=ccrs.Robinson(central_longitude=0); pc=ccrs.PlateCarree()
levs=[("850 hPa", find_k(pPa,850)), ("500 hPa", find_k(pPa,500)), ("250 hPa", find_k(pPa,250))]
ncols = 3 if pv_sig is None else 4
fig, axes = plt.subplots(3, ncols, figsize=(6*ncols, 14), subplot_kw={"projection": proj})
axes=np.atleast_2d(axes)

for row, (name, k) in enumerate(levs):
    data_cols = [
        (pv_e2[k], f"ERA5 PV\n{name}"),
        (pv_sh[k], f"SH PV\n{name}"),
        (pv_sh[k]-pv_e2[k], f"Diff SH−ERA5\n{name}"),
    ]
    if pv_sig is not None:
        ks = np.argmin(np.abs(pPa/100000.0 - (float(name.split()[0])/1000)))
        data_cols.append((pv_sig[ks]-pv_era5_sig[ks], f"Diff σ−ERA5\n{name}"))
    
    for col, (data, title) in enumerate(data_cols):
        ax=axes[row,col]; is_diff="Diff" in title
        vm=np.nanpercentile(np.abs(data),99)
        if is_diff: vm=max(vm,0.1); cmap="RdBu_r"
        else: cmap="RdBu_r"
        ax.set_global(); ax.add_feature(cfeature.COASTLINE,lw=0.3,edgecolor="0.3")
        cf=ax.pcolormesh(lon,lat_sh,data,cmap=cmap,transform=pc,vmin=-vm,vmax=vm,rasterized=True)
        plt.colorbar(cf,ax=ax,shrink=0.6,pad=0.02,label="PVU" if not is_diff else "Δ PVU")
        ax.set_title(title,fontsize=9)

fig.suptitle("Ertel PV: SH Gradients vs ERA5 (2025-01-08 00Z)",fontsize=13,y=0.98)
plt.tight_layout()
out=PLOT_DIR/"era5_pv_sh_comparison.png"
fig.savefig(out,dpi=150,bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close(fig)

# RMS profile
fig2,ax2=plt.subplots(figsize=(7,8))
rms_sh=np.sqrt(np.nanmean((pv_sh-pv_e2)**2,axis=(1,2)))
ax2.plot(rms_sh,pPa/100,"b-o",label="SH PV",markersize=4)
ax2.set_ylim(pPa[-1]/100,pPa[0]/100)
ax2.set_ylabel("Pressure [hPa]"); ax2.set_xlabel("RMS diff [PVU]")
ax2.set_title("Vertical Profile: SH PV − ERA5 PV (2025-01-08 00Z)")
ax2.legend(); ax2.grid(True,alpha=0.3)
plt.tight_layout()
out2=PLOT_DIR/"era5_pv_sh_rms_profile.png"
fig2.savefig(out2,dpi=150,bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)
print("\nDone.")
