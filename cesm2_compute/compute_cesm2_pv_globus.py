#!/usr/bin/env python
"""Ertel PV from NATIVE MODEL-LEVEL (hybrid) CESM2-LE data staged via Globus.

Reads CAM history files (U,V,T on 32 hybrid sigma-pressure levels + PS) that
embed the hybrid coefficients ``hyam/hybm/P0``, reconstructs the TRUE pressure
of each model level

    p(k,j,i) = hyam(k) * P0 + hybm(k) * PS(j,i),

and computes Ertel PV on the fixed sigma grid via the hybrid-aware
``ertel_pv_sigma`` (see src/ertel_pv.py).  This is the correct treatment that
the AWS-zarr path could not do (LENS2 zarr drops the coefficients and exposes
only the NOMINAL `lev` = (hyam+hybm)*P0).

Single-date sample + validation plot per member.

Usage
-----
    micromamba run -n blocking python compute_cesm2_pv_globus.py \
        --stage-dir /net/flood/home/x_yan/cmip6/d651056/modellevel_m1_d20102014 \
        --member 1 --date 2010-01-01 [--cleanup]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
from src.ertel_pv import ertel_pv_sigma, DEFAULT_SIGMA_LEVELS


def _find_file(stage_dir: Path, var: str):
    """Locate the single CAM history file for a variable in the stage dir."""
    hits = sorted(stage_dir.glob(f"*.cam.h*.{var}.*.nc"))
    if not hits:
        raise FileNotFoundError(f"No {var} file in {stage_dir}")
    return hits[0]


def _date_index(ds, ymd):
    """Index of the timestep matching (year, month, day)."""
    y, m, d = ymd
    times = ds.time.values
    for i, t in enumerate(times):
        if (t.year, t.month, t.day) == (y, m, d):
            return i
    raise ValueError(f"date {ymd} not found (range {times[0]} .. {times[-1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True, type=Path)
    ap.add_argument("--member", required=True, type=int)
    ap.add_argument("--date", default="2010-01-01", help="YYYY-MM-DD sample date")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "globus_out")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the large raw staged .nc after computing")
    args = ap.parse_args()

    ymd = tuple(int(x) for x in args.date.split("-"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "plots"; plot_dir.mkdir(exist_ok=True)

    f_t = _find_file(args.stage_dir, "T")
    f_u = _find_file(args.stage_dir, "U")
    f_v = _find_file(args.stage_dir, "V")
    f_ps = _find_file(args.stage_dir, "PS")

    # Lazy open (netCDF4 reads only the requested slice into memory).
    ds_t = xr.open_dataset(f_t)
    ds_u = xr.open_dataset(f_u)
    ds_v = xr.open_dataset(f_v)
    ds_ps = xr.open_dataset(f_ps)

    # Hybrid coefficients + reference pressure (embedded in the model-level file).
    for c in ("hyam", "hybm", "P0", "lev"):
        if c not in ds_t.variables:
            raise KeyError(f"{c} missing from {f_t.name} — not a model-level file?")
    hyam = ds_t["hyam"].values.astype(float)
    hybm = ds_t["hybm"].values.astype(float)
    P0 = float(ds_t["P0"].values)
    lev = ds_t["lev"].values.astype(float)
    nlev = lev.size
    print(f"  model levels: {nlev}  | P0 = {P0:.0f} Pa")
    print(f"  lev (nominal hPa): {lev[0]:.2f} (top) .. {lev[-1]:.2f} (sfc)")
    print(f"  hybm: top={hybm[0]:.4f} (pure-pressure)  sfc={hybm[-1]:.4f} (near-sigma)")

    it = _date_index(ds_t, ymd)
    iu = _date_index(ds_u, ymd)
    iv = _date_index(ds_v, ymd)
    ips = _date_index(ds_ps, ymd)
    actual = ds_t.time.values[it]
    print(f"  sample date: {actual}")

    lat = ds_t.lat.values.astype(float)
    lon = ds_t.lon.values.astype(float)
    t_arr = ds_t["T"].isel(time=it).values.astype(float)
    u_arr = ds_u["U"].isel(time=iu).values.astype(float)
    v_arr = ds_v["V"].isel(time=iv).values.astype(float)
    ps_arr = ds_ps["PS"].isel(time=ips).values.astype(float)
    print(f"  shapes: T{t_arr.shape}  PS{ps_arr.shape}  "
          f"PS range [{ps_arr.min():.0f}, {ps_arr.max():.0f}] Pa")

    # CAM order is top->surface. Flip to surface->top and keep hyam/hybm aligned.
    u_arr = u_arr[::-1]; v_arr = v_arr[::-1]; t_arr = t_arr[::-1]
    hyam_s = hyam[::-1]; hybm_s = hybm[::-1]

    # True pressure check (bottom level should ~ PS; top ~ hyam*P0)
    p_bot = hyam_s[0] * P0 + hybm_s[0] * ps_arr
    print(f"  true p(bottom level): [{p_bot.min():.0f}, {p_bot.max():.0f}] Pa "
          f"(≈ PS, NOT the nominal {lev[-1]*100:.0f} Pa)")

    # ---- hybrid-correct sigma PV ----
    pv_sigma, p_s3d = ertel_pv_sigma(
        u_arr, v_arr, t_arr, None, ps_arr, lat, lon,
        hyam=hyam_s, hybm=hybm_s, p0=P0,
    )
    print(f"  PV sigma range: [{np.nanmin(pv_sigma):.1f}, {np.nanmax(pv_sigma):.1f}] PVU"
          f"  | NaN={100*np.isnan(pv_sigma).mean():.2f}%")

    # ---- save PV netCDF (to data2 out dir) ----
    sig = DEFAULT_SIGMA_LEVELS
    date_str = f"{actual.year:04d}-{actual.month:02d}-{actual.day:02d}"
    out_nc = args.out_dir / f"cesm2le_m{args.member:02d}_pv_sigma_{date_str}.nc"
    xr.Dataset(
        {"pv_sigma": (["sigma", "lat", "lon"], pv_sigma),
         "p_sigma":  (["sigma", "lat", "lon"], p_s3d),
         "ps":       (["lat", "lon"], ps_arr)},
        coords={"sigma": sig, "lat": lat, "lon": lon},
        attrs={"member": args.member, "date": date_str,
               "method": "hybrid model-level p=hyam*P0+hybm*PS",
               "source": "CESM2-LE GDEX d651056 native cam.h6 (Globus)"},
    ).to_netcdf(out_nc)
    print(f"  wrote {out_nc}")

    # ---- validation plot ----
    proj = ccrs.Robinson(central_longitude=0); pc = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
    for ax, st in zip(axes, [0.85, 0.50, 0.25]):
        k = int(np.argmin(np.abs(sig - st)))
        d = pv_sigma[k]
        vm = np.nanpercentile(np.abs(d), 99)
        ax.set_global(); ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
        cf = ax.pcolormesh(lon, lat, d, cmap="RdBu_r", transform=pc,
                           vmin=-vm, vmax=vm, rasterized=True)
        plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02, label="PVU")
        ax.set_title(f"σ={sig[k]:.3f}  (nom {sig[k]*1013:.0f} hPa)", fontsize=9)
    fig.suptitle(f"CESM2-LE m{args.member:02d} Ertel PV — hybrid model-level "
                 f"(Globus d651056, {date_str})", fontsize=13, y=0.99)
    plt.tight_layout()
    out_png = plot_dir / f"cesm2le_m{args.member:02d}_pv_sigma_{date_str}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out_png}")

    for ds in (ds_t, ds_u, ds_v, ds_ps):
        ds.close()

    if args.cleanup:
        for f in (f_u, f_v, f_t, f_ps):
            try:
                f.unlink(); print(f"  removed raw {f.name}")
            except OSError as e:
                print(f"  could not remove {f}: {e}")

    print(f"Done: member {args.member}, {date_str}")


if __name__ == "__main__":
    main()
