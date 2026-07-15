#!/usr/bin/env python
"""Step 03 — interpolate the σ-level Ertel PV onto ISOBARIC and/or ISENTROPIC surfaces.

Reads the σ-coordinate PV netCDF written by step 02 (``02_compute_pv_hybrid/``),
which carries ``pv_sigma``, ``p_sigma`` (actual pressure of each σ level) and
``theta_sigma`` (θ on σ), and samples the PV onto:

* **pressure** levels — per-column log-p interp (``interp_to_pressure``),
* **isentropic** θ levels — per-column interp using θ(σ) as the vertical
  coordinate (``interp_to_isentropic``; default {300,315,320,330,350} K).

Both interpolators live in ``src/ertel_pv.py`` and are shared with the ERA5
sanity check (``test00``). Follows MPAS best-practice: PV is computed on the
native σ grid (step 02) and only then mapped to output surfaces — never by
interpolating u/v/T to pressure first.

Usage
-----
    micromamba run -n blocking python interp_pv_to_levels.py \
        --pv-sigma-nc ../02_compute_pv_hybrid/out/cesm2le_m01_pv_sigma_2010-01-01.nc \
        --output-coords pressure,isentropic
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
from src.ertel_pv import interp_to_pressure, interp_to_isentropic, DEFAULT_THETA_LEVELS

# default isobaric output levels [hPa]
DEFAULT_PRESSURE_HPA = np.array([1000., 850., 700., 500., 300., 250., 200., 100.])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-sigma-nc", required=True, type=Path,
                    help="σ-PV netCDF from step 02 (pv_sigma/p_sigma/theta_sigma/ps)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--output-coords", default="pressure,isentropic",
                    help="comma list: any of pressure,isentropic (default: both)")
    ap.add_argument("--theta-levels", default=None,
                    help="comma list of isentropic θ levels [K] "
                         "(default 300,315,320,330,350 — RWB/blocking set)")
    ap.add_argument("--pressure-levels", default=None,
                    help="comma list of isobaric output levels [hPa] "
                         "(default 1000,850,700,500,300,250,200,100)")
    args = ap.parse_args()

    coords = [c.strip() for c in args.output_coords.split(",") if c.strip()]
    bad = set(coords) - {"pressure", "isentropic"}
    if bad:
        ap.error(f"unknown --output-coords {sorted(bad)}; choose from pressure,isentropic")
    theta_levels = (np.array([float(x) for x in args.theta_levels.split(",")])
                    if args.theta_levels else DEFAULT_THETA_LEVELS)
    pressure_hpa = (np.array([float(x) for x in args.pressure_levels.split(",")])
                    if args.pressure_levels else DEFAULT_PRESSURE_HPA)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "plots"; plot_dir.mkdir(exist_ok=True)

    # ---- read the σ-PV hand-off from step 02 ----
    ds = xr.open_dataset(args.pv_sigma_nc)
    pv_sigma = ds["pv_sigma"].values
    p_sigma = ds["p_sigma"].values
    theta_sigma = ds["theta_sigma"].values
    lat = ds["lat"].values.astype(float)
    lon = ds["lon"].values.astype(float)
    member = int(ds.attrs.get("member", 0))
    date_str = str(ds.attrs.get("date", "unknown"))
    print(f"  read {args.pv_sigma_nc}  (member {member}, {date_str}, "
          f"pv_sigma {pv_sigma.shape})")

    pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=0)

    def _plot(field3d, levs, show, fmt, tag):
        levs = np.asarray(levs, dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})
        for ax, tgt in zip(np.atleast_1d(axes), show):
            k = int(np.argmin(np.abs(levs - tgt)))
            d = field3d[k]; valid = np.isfinite(d)
            vm = np.nanpercentile(np.abs(d[valid]), 99) if valid.any() else 1.0
            ax.set_global(); ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
            cf = ax.pcolormesh(lon, lat, d, cmap="RdBu_r", transform=pc,
                               vmin=-vm, vmax=vm, rasterized=True)
            plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02, label="PVU")
            ax.set_title(fmt(levs[k]), fontsize=9)
        fig.suptitle(f"CESM2-LE m{member:02d} Ertel PV — {tag} (from σ-PV, {date_str})",
                     fontsize=13, y=0.99)
        plt.tight_layout()
        out_png = plot_dir / f"cesm2le_m{member:02d}_pv_{tag}_{date_str}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out_png}")

    common_attrs = {"member": member, "date": date_str,
                    "source": str(args.pv_sigma_nc.name),
                    "note": "σ-PV interpolated to output surface (ζ evaluated on σ)"}
    print(f"  output coordinates: {coords}")

    # ---- pressure (isobaric) ----
    if "pressure" in coords:
        pv_p = interp_to_pressure(pv_sigma, p_sigma, pressure_hpa * 100.0)
        print(f"  PV pressure range: [{np.nanmin(pv_p):.1f}, {np.nanmax(pv_p):.1f}] PVU")
        out_nc = args.out_dir / f"cesm2le_m{member:02d}_pv_pressure_{date_str}.nc"
        xr.Dataset({"pv_pressure": (["plev", "lat", "lon"], pv_p)},
                   coords={"plev": pressure_hpa, "lat": lat, "lon": lon},
                   attrs=common_attrs).to_netcdf(out_nc)
        print(f"  wrote {out_nc}")
        _plot(pv_p, pressure_hpa, [850., 500., 250.], lambda p: f"{p:.0f} hPa", "pressure")

    # ---- isentropic (θ) ----
    if "isentropic" in coords:
        pv_th = interp_to_isentropic(pv_sigma, theta_sigma, theta_levels)
        print(f"  PV isentropic range: [{np.nanmin(pv_th):.1f}, {np.nanmax(pv_th):.1f}] PVU")
        out_nc = args.out_dir / f"cesm2le_m{member:02d}_pv_isentropic_{date_str}.nc"
        xr.Dataset({"pv_theta": (["theta", "lat", "lon"], pv_th)},
                   coords={"theta": theta_levels, "lat": lat, "lon": lon},
                   attrs=common_attrs).to_netcdf(out_nc)
        print(f"  wrote {out_nc}")
        show_th = [theta_levels[min(1, len(theta_levels) - 1)],
                   theta_levels[len(theta_levels) // 2], theta_levels[-1]]
        _plot(pv_th, theta_levels, show_th, lambda th: f"θ={th:.0f} K", "isentropic")

    ds.close()
    print(f"Done: member {member}, {date_str}")


if __name__ == "__main__":
    main()
