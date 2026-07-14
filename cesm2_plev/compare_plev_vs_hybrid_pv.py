#!/usr/bin/env python3
"""
Phase-0 GATE: pressure-level vs hybrid-native Ertel PV comparison (CESM2-LENS2).

Computes Ertel PV two ways for the SAME member/day(s) and compares at 850/500/250 hPa:
  • HYBRID  : native h6 U,V,T (32 model levels) + PS, true p = hyam*P0 + hybm*PS
              -> ertel_pv_sigma(hyam,hybm,p0) -> interp sigma -> pressure   [reference]
  • PLEV    : day_1 pressure-level U,V,T on 8 levels [10,50,100,200,500,700,850,1000] + PS
              -> ertel_pv_sigma(plev=INPUT_PA)   -> interp sigma -> pressure   [candidate]

850 & 500 are NATIVE plev levels; 250 is INTERPOLATED on the plev side (from 200/500) — the
decisive UTLS test. Reports per level: below-ground NaN-mask agreement + Pearson r over common
valid points, and writes hybrid/plev/diff PV maps for a vision check.

GATE to adopt the cheaper plev path: r >= 0.95 at ALL of 850/500/250 AND NaN masks agree.

Usage:
    micromamba run -n blocking python compare_plev_vs_hybrid_pv.py \
        --data-dir /net/flood/data2/users/x_yan/pv_ertel_compute/cesm2_plev/test_m1_2010 \
        --member 1 --dates 2010-01-15,2010-07-15,2011-04-15,2012-10-15,2013-07-16,2014-01-15
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

_PV_ROOT = Path("/net/flood/data2/users/x_yan/pv_ertel_compute")
if str(_PV_ROOT) not in sys.path:
    sys.path.insert(0, str(_PV_ROOT))
from src.ertel_pv import ertel_pv_sigma, interp_to_pressure  # noqa: E402

INPUT_HPA = np.array([10., 50., 100., 200., 500., 700., 850., 1000.], dtype=float)  # native plev
INPUT_PA = INPUT_HPA * 100.0
TARGET_HPA = np.array([850., 500., 250.], dtype=float)   # comparison levels (250 = interpolated)
TARGET_PA = TARGET_HPA * 100.0
R_GATE = 0.95


# interp_to_pressure now lives in src.ertel_pv (imported above) so the σ→pressure
# path is shared with the main pipeline and the ERA5 sanity check.


def _find(data_dir, stream, var):
    hits = sorted(Path(data_dir).glob(f"*.cam.{stream}.{var}.*.nc"))
    if not hits:
        raise FileNotFoundError(f"no {stream}.{var} in {data_dir}")
    return hits[0]


def _date_idx(ds, ymd):
    for i, t in enumerate(ds.time.values):
        if (t.year, t.month, t.day) == ymd:
            return i
    return None


def hybrid_pv(dh, ymd, lat, lon):
    """Ertel PV at TARGET from native hybrid U,V,T,PS for one day (top->sfc flipped to sfc->top)."""
    it = _date_idx(dh["T"], ymd)
    if it is None:
        return None
    hyam = dh["T"]["hyam"].values.astype(float)[::-1]
    hybm = dh["T"]["hybm"].values.astype(float)[::-1]
    P0 = float(dh["T"]["P0"].values)
    t = dh["T"]["T"].isel(time=it).values.astype(float)[::-1]
    u = dh["U"]["U"].isel(time=_date_idx(dh["U"], ymd)).values.astype(float)[::-1]
    v = dh["V"]["V"].isel(time=_date_idx(dh["V"], ymd)).values.astype(float)[::-1]
    ps = dh["PS"]["PS"].isel(time=_date_idx(dh["PS"], ymd)).values.astype(float)
    pv_s, p_s3d = ertel_pv_sigma(u, v, t, None, ps, lat, lon,
                                 hyam=hyam, hybm=hybm, p0=P0, method="full")
    return interp_to_pressure(pv_s, p_s3d, TARGET_PA)


def plev_pv(dp, ymd, lat, lon):
    """Ertel PV at TARGET from 8-level pressure U,V,T,PS for one day."""
    nlat, nlon = len(lat), len(lon)
    u = np.stack([dp[f"U{p}"][f"U{p}"].isel(time=_date_idx(dp[f"U{p}"], ymd)).values
                  for p in PL_STRS]).astype(float)
    v = np.stack([dp[f"V{p}"][f"V{p}"].isel(time=_date_idx(dp[f"V{p}"], ymd)).values
                  for p in PL_STRS]).astype(float)
    t = np.stack([dp[f"T{p}"][f"T{p}"].isel(time=_date_idx(dp[f"T{p}"], ymd)).values
                  for p in PL_STRS]).astype(float)
    ps = dp["PS"]["PS"].isel(time=_date_idx(dp["PS"], ymd)).values.astype(float)
    pv_s, p_s3d = ertel_pv_sigma(u, v, t, INPUT_PA, ps, lat, lon, method="full")
    return interp_to_pressure(pv_s, p_s3d, TARGET_PA)


PL_STRS = ["010", "050", "100", "200", "500", "700", "850", "1000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--member", type=int, default=1)
    ap.add_argument("--dates", required=True, help="comma list YYYY-MM-DD")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    dd = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dd
    dates = [tuple(int(x) for x in d.split("-")) for d in args.dates.split(",")]

    # open lazily
    dh = {v: xr.open_dataset(_find(dd, "h6", v)) for v in ("U", "V", "T")}
    dh["PS"] = xr.open_dataset(_find(dd, "h1", "PS"))
    dp = {f"{b}{p}": xr.open_dataset(_find(dd, "h1", f"{b}{p}"))
          for b in ("U", "V", "T") for p in PL_STRS}
    dp["PS"] = dh["PS"]
    lat = dh["T"].lat.values.astype(float); lon = dh["T"].lon.values.astype(float)
    print(f"grid {len(lat)}x{len(lon)}  | dates {dates}")

    # accumulate per-level stats over all days
    acc = {k: {"hyb": [], "plv": []} for k in TARGET_HPA}
    first = None
    for ymd in dates:
        hp = hybrid_pv(dh, ymd, lat, lon)
        pp = plev_pv(dp, ymd, lat, lon)
        if hp is None or pp is None:
            print(f"  {ymd}: date not found — skip"); continue
        if first is None:
            first = (ymd, hp, pp)
        for k, lev in enumerate(TARGET_HPA):
            acc[lev]["hyb"].append(hp[k].ravel())
            acc[lev]["plv"].append(pp[k].ravel())
        print(f"  {ymd}: done")

    print("\n=== Per-level comparison (pooled over days) ===")
    gate_ok = True
    for lev in TARGET_HPA:
        h = np.concatenate(acc[lev]["hyb"]); p = np.concatenate(acc[lev]["plv"])
        hb_nan = np.isnan(h); pl_nan = np.isnan(p)
        nan_agree = 100.0 * np.mean(hb_nan == pl_nan)
        both = (~hb_nan) & (~pl_nan)
        r = np.corrcoef(h[both], p[both])[0, 1] if both.sum() > 10 else np.nan
        bias = float(np.mean(p[both] - h[both]))
        rmse = float(np.sqrt(np.mean((p[both] - h[both]) ** 2)))
        native = "native" if lev in (850., 500.) else "INTERPOLATED"
        ok = (r >= R_GATE) and (nan_agree > 99.0)
        gate_ok &= ok
        print(f"  {lev:4.0f} hPa ({native:11s}): r={r:.4f}  NaN-agree={nan_agree:.2f}%  "
              f"bias={bias:+.3f} rmse={rmse:.3f} PVU  valid={both.sum():,}  "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\n=== GATE: {'PLEV OK (r>=0.95 & NaN match at all levels)' if gate_ok else 'USE HYBRID (a level failed)'} ===")

    # vision-check figure for the first day: hybrid / plev / diff, 3 levels
    if first is not None:
        ymd, hp, pp = first
        ds = f"{ymd[0]:04d}-{ymd[1]:02d}-{ymd[2]:02d}"
        pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=0)
        fig, axes = plt.subplots(3, 3, figsize=(18, 11), subplot_kw={"projection": proj})
        for r_i, lev in enumerate(TARGET_HPA):
            fields = [hp[r_i], pp[r_i], pp[r_i] - hp[r_i]]
            titles = [f"hybrid {lev:.0f}", f"plev {lev:.0f}", f"plev−hybrid {lev:.0f}"]
            vm = np.nanpercentile(np.abs(hp[r_i]), 99)
            for c_i, (fld, ttl) in enumerate(zip(fields, titles)):
                ax = axes[r_i, c_i]; ax.set_global()
                ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.3")
                vmax = vm if c_i < 2 else max(0.5, np.nanpercentile(np.abs(fields[2]), 99))
                cf = ax.pcolormesh(lon, lat, fld, cmap="RdBu_r", transform=pc,
                                   vmin=-vmax, vmax=vmax, rasterized=True)
                plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02, label="PVU")
                ax.set_title(ttl, fontsize=10)
        fig.suptitle(f"CESM2-LE m{args.member:02d} Ertel PV: hybrid vs plev ({ds})", y=0.99, fontsize=13)
        plt.tight_layout()
        out = out_dir / f"compare_pv_plev_vs_hybrid_m{args.member:02d}_{ds}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
