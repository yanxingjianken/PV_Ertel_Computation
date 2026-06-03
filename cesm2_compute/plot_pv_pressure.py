#!/usr/bin/env python
"""Ertel PV on PRESSURE levels (850/500/250 hPa) from native hybrid CESM2 data.

Pipeline: native hybrid U/V/T/PS  ->  sigma-coordinate PV (true p=hyam*P0+hybm*PS)
          ->  interpolate PV from sigma surfaces to fixed pressure levels  ->  plot.

PV is computed on terrain-following sigma levels (best near surface), then mapped to
isobaric surfaces for display.  Targets below the local surface (e.g. 850 hPa over high
terrain) are NaN by construction — physically correct (underground).

Usage:
    micromamba run -n blocking python plot_pv_pressure.py \
        --stage-dir globus_data/sample_m01_2010-01-01 --member 1 --date 2010-01-01
"""
import argparse, sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))
from src.ertel_pv import ertel_pv_sigma

PLEVELS_HPA = np.array([850.0, 500.0, 250.0])   # plot targets


def _find(stage, var):
    hits = sorted(Path(stage).glob(f"*.cam.h*.{var}.*.nc"))
    if not hits:
        raise FileNotFoundError(f"no {var} in {stage}")
    return hits[0]


def _date_idx(ds, ymd):
    for i, t in enumerate(ds.time.values):
        if (t.year, t.month, t.day) == ymd:
            return i
    raise ValueError(f"{ymd} not in {ds.time.values[0]}..{ds.time.values[-1]}")


def interp_sigma_to_pressure(field_sig, p_sig3d, target_pa):
    """Per-column log-p interp of a sigma-coordinate field onto pressure levels.

    field_sig, p_sig3d : (nsig, nlat, nlon) — p_sig3d is the ACTUAL pressure
        (= sigma*ps) at each sigma surface.  target_pa : (ntgt,) [Pa].
    Out-of-column-range targets -> NaN.
    """
    nsig, nlat, nlon = field_sig.shape
    ntgt = len(target_pa)
    out = np.full((ntgt, nlat, nlon), np.nan, dtype=float)
    logt = np.log(target_pa)
    for j in range(nlat):
        for i in range(nlon):
            p = p_sig3d[:, j, i]; f = field_sig[:, j, i]
            if p[0] > p[-1]:                 # need ascending pressure
                p = p[::-1]; f = f[::-1]
            out[:, j, i] = np.interp(logt, np.log(p), f,
                                     left=np.nan, right=np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--member", type=int, required=True)
    ap.add_argument("--date", default="2010-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    ymd = tuple(int(x) for x in args.date.split("-"))

    ds_t = xr.open_dataset(_find(args.stage_dir, "T"))
    ds_u = xr.open_dataset(_find(args.stage_dir, "U"))
    ds_v = xr.open_dataset(_find(args.stage_dir, "V"))
    ds_ps = xr.open_dataset(_find(args.stage_dir, "PS"))

    hyam = ds_t["hyam"].values.astype(float)[::-1]   # flip top->sfc => sfc->top
    hybm = ds_t["hybm"].values.astype(float)[::-1]
    P0 = float(ds_t["P0"].values)
    lat = ds_t.lat.values.astype(float); lon = ds_t.lon.values.astype(float)

    it = _date_idx(ds_t, ymd)
    t = ds_t["T"].isel(time=it).values.astype(float)[::-1]
    u = ds_u["U"].isel(time=_date_idx(ds_u, ymd)).values.astype(float)[::-1]
    v = ds_v["V"].isel(time=_date_idx(ds_v, ymd)).values.astype(float)[::-1]
    ps = ds_ps["PS"].isel(time=_date_idx(ds_ps, ymd)).values.astype(float)

    pv_sig, p_sig3d = ertel_pv_sigma(u, v, t, None, ps, lat, lon,
                                     hyam=hyam, hybm=hybm, p0=P0)
    pv_p = interp_sigma_to_pressure(pv_sig, p_sig3d, PLEVELS_HPA * 100.0)
    for k, pl in enumerate(PLEVELS_HPA):
        d = pv_p[k]
        print(f"  {pl:.0f} hPa: PV [{np.nanmin(d):.1f}, {np.nanmax(d):.1f}] PVU, "
              f"NaN(below-ground)={100*np.isnan(d).mean():.1f}%")

    date_str = f"{ymd[0]:04d}-{ymd[1]:02d}-{ymd[2]:02d}"
    proj = ccrs.Robinson(central_longitude=0); pc = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
    for ax, pl, d in zip(axes, PLEVELS_HPA, pv_p):
        vm = np.nanpercentile(np.abs(d), 99)
        ax.set_global(); ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
        cf = ax.pcolormesh(lon, lat, d, cmap="RdBu_r", transform=pc,
                           vmin=-vm, vmax=vm, rasterized=True)
        plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02, label="PVU")
        ax.set_title(f"{pl:.0f} hPa", fontsize=11)
    fig.suptitle(f"CESM2-LE m{args.member:02d} Ertel PV on pressure levels "
                 f"(hybrid model-level source, {date_str})", fontsize=13, y=0.99)
    plt.tight_layout()
    out = args.out_dir / f"cesm2le_m{args.member:02d}_pv_pressure_850_500_250_{date_str}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
