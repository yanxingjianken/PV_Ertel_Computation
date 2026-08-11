#!/usr/bin/env python3
"""test02 step 0 — p(theta) geometry check against JRA-3Q.

The cheapest and most localising test in the suite: it uses only the hybrid
coefficients, surface pressure and temperature, and compares the pressure of each
isentrope against JRA-3Q's own `anl_isentrop/pres`.  No PV, no spherical
harmonics, so a failure here can only come from the vertical coordinate -- which
is exactly the piece that differs between JRA and CAM:

    JRA-3Q :  p = a + b*ps          a in Pa,          levels surface -> top
    CAM    :  p = hyam*P0 + hybm*ps hyam dimensionless, levels top -> surface

Getting `a` wrong fails loudly (~1e9 Pa).  Getting the level ORDER wrong fails
silently, because the interpolators only require monotonicity, not a direction --
a flipped profile still interpolates, just to the wrong answer.  This step is what
catches that.

Run:  python step00_pressure_geometry.py [--date 2015-01-05T06]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cftime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ertel_pv import KAPPA, P0                     # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

JRA = "/glade/campaign/collections/rda/data/d640000"
OUT = os.path.dirname(os.path.abspath(__file__))
THETA_CHECK = np.array([300., 310., 320., 330., 340., 350.])


def _masked_to_nan(x):
    """netCDF4 hands back masked arrays; `np.asarray` on one exposes the RAW FILL
    VALUE (~1e20) as if it were data.  JRA's `pres-theta` is missing wherever the
    isentrope is below ground -- common at 300/310 K over terrain, never at 320 K
    and above -- so a naive conversion produced 1e15 hPa "biases" at exactly the
    two lowest theta levels while everything above looked perfect.  Convert
    explicitly."""
    return np.ma.filled(np.ma.masked_invalid(np.ma.asarray(x)).astype(np.float64), np.nan)


def _open_at(pattern, varname, when):
    """Open the file in `pattern` covering `when` and return (var, time index)."""
    for f in sorted(glob.glob(pattern)):
        d = nc.Dataset(f)
        tv = d.variables["time"]
        dates = cftime.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
        hit = np.nonzero(np.array([str(x)[:13] for x in dates]) == str(when)[:13])[0]
        if hit.size:
            return d, int(hit[0]), f
        d.close()
    raise SystemExit(f"{when} not found in {pattern}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2015-01-05 06")
    args = ap.parse_args()
    when = args.date.replace("T", " ")
    ym = when[:4] + when[5:7]

    d_t, k_t, f_t = _open_at(f"{JRA}/anl_mdl/{ym}/*tmp-hyb-an-gauss*", "tmp", when)
    d_p, k_p, f_p = _open_at(f"{JRA}/anl_surf/{ym}/*pres-sfc-an-gauss*", "pres", when)
    d_i, k_i, f_i = _open_at(f"{JRA}/anl_isentrop/{ym}/*pres-theta-an-gauss*", "pres", when)

    a = np.asarray(d_t.variables["a_hybrid_level"][:], dtype=np.float64)    # Pa
    b = np.asarray(d_t.variables["b_hybrid_level"][:], dtype=np.float64)    # 1
    T = _masked_to_nan(d_t.variables["tmp-hyb-an-gauss"][k_t])
    ps = _masked_to_nan(d_p.variables["pres-sfc-an-gauss"][k_p])
    lat = np.asarray(d_t.variables["lat"][:], dtype=np.float64)

    th_jra = np.asarray(d_i.variables["isentropic_level"][:], dtype=np.float64)
    p_jra = _masked_to_nan(d_i.variables["pres-theta-an-gauss"][k_i])

    print(f"time      : {when}")
    print(f"hybrid    : {len(a)} levels, a[0]={a[0]:.3f} Pa b[0]={b[0]:.4f} "
          f"-> {'surface' if b[0] > b[-1] else 'top'} first")
    print(f"grid      : {T.shape}   ps {ps.shape}   JRA p(theta) {p_jra.shape}")

    # JRA convention -- NOT the CAM one
    p3d = a[:, None, None] + b[:, None, None] * ps[None, :, :]
    theta = T * (P0 / p3d) ** KAPPA

    # interpolate pressure onto the theta surfaces, per column, in log p
    sys.path.insert(0, "/glade/derecho/scratch/kenyan/01_cesm_processing/06_isentropic_clim")
    from pv_isentropic import interp_monotonic
    idx = [int(np.argmin(np.abs(th_jra - t))) for t in THETA_CHECK]
    p_ours = interp_monotonic(p3d, theta, THETA_CHECK, log=True)

    print(f"\n{'theta[K]':>9} {'n valid':>9} {'bias [hPa]':>12} {'RMSE [hPa]':>12} "
          f"{'rel RMSE':>10} {'corr':>7}")
    rows = []
    for n, t in enumerate(THETA_CHECK):
        ours = p_ours[n] / 100.0
        ref = p_jra[idx[n]] / 100.0
        ok = np.isfinite(ours) & np.isfinite(ref) & (ref > 0) & (ref < 1200.0)
        if ok.sum() < 100:
            print(f"{t:9.0f} {ok.sum():9d}   (too few valid points)")
            continue
        dif = ours[ok] - ref[ok]
        rmse = float(np.sqrt(np.mean(dif ** 2)))
        corr = float(np.corrcoef(ours[ok], ref[ok])[0, 1])
        rel = rmse / float(np.mean(ref[ok]))
        rows.append((t, ok.sum(), dif.mean(), rmse, rel, corr))
        print(f"{t:9.0f} {ok.sum():9d} {dif.mean():12.3f} {rmse:12.3f} "
              f"{100*rel:9.2f}% {corr:7.4f}")

    # figure
    fig, axes = plt.subplots(3, len(THETA_CHECK), figsize=(3.1 * len(THETA_CHECK), 8.4),
                             constrained_layout=True)
    for n, t in enumerate(THETA_CHECK):
        ours = p_ours[n] / 100.0
        ref = p_jra[idx[n]] / 100.0
        vmin, vmax = np.nanpercentile(ref, [2, 98])
        for r, (fld, ttl, cmap, lim) in enumerate((
                (ref, "JRA-3Q  p(θ)", "viridis", (vmin, vmax)),
                (ours, "ours  p(θ)", "viridis", (vmin, vmax)),
                (ours - ref, "ours − JRA", "RdBu_r", None))):
            ax = axes[r, n]
            kw = dict(cmap=cmap)
            if lim:
                kw.update(vmin=lim[0], vmax=lim[1])
            else:
                m = np.nanpercentile(np.abs(fld), 99)
                kw.update(vmin=-m, vmax=m)
            im = ax.pcolormesh(np.arange(fld.shape[1]), lat, fld, shading="auto", **kw)
            if r == 0:
                ax.set_title(f"θ = {t:.0f} K", fontsize=11)
            if n == 0:
                ax.set_ylabel(f"{ttl}\nlat [°]", fontsize=9)
            ax.set_xticks([])
            if r == 2:
                fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.02,
                             label="Δp [hPa]")
            elif n == len(THETA_CHECK) - 1:
                fig.colorbar(im, ax=ax, label="p [hPa]")
    fig.suptitle(f"test02 step 0 — pressure of each isentrope, ours vs JRA-3Q  ({when})\n"
                 f"JRA hybrid convention p = a + b·ps  (a in Pa, levels surface→top)",
                 fontsize=12)
    png = os.path.join(OUT, "step00_pressure_geometry.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"\nwrote {png}")

    worst = max((r[4] for r in rows), default=np.inf)
    print(f"\nGATE: worst relative RMSE = {100*worst:.2f}%  "
          f"{'PASS' if worst < 0.02 else 'FAIL — check the hybrid convention/level order'}")
    for d in (d_t, d_p, d_i):
        d.close()


if __name__ == "__main__":
    main()
