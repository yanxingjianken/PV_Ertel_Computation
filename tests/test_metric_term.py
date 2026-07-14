#!/usr/bin/env python
"""Proof that the spherical vorticity in this repo keeps the ``+u·tanφ/a`` metric term.

On a lat-lon grid, relative vorticity is

    ζ = (1/(a cosφ)) [ ∂v/∂λ − ∂(u cosφ)/∂φ ]
      = ∂v/∂x − ∂u/∂y + u·tanφ/a          (∂/∂x ≡ (1/(a cosφ))∂/∂λ, ∂/∂y ≡ (1/a)∂/∂φ)

A naive *flat* finite difference computes only ``∂v/∂x − ∂u/∂y`` and silently drops the
metric term ``+u·tanφ/a``. This repo computes ζ with the spherical-harmonic operator
``pvtend.sh_ops.vortdiv_sh`` (pyspharm ``getvrtdivspec``), which evaluates the exact
spherical curl — so the metric term is analytically included and nothing is missing.

Test field: solid-body rotation ``u = U0 cosφ, v = 0`` → exact ``ζ = 2 U0 sinφ / a``.
For this field the flat FD gives exactly *half* the truth; the metric term supplies the
other half. We assert:
  1. spherical FD (flat + u·tanφ/a) reproduces the analytic ζ,
  2. the SH operator reproduces the analytic ζ,
  3. the flat FD (metric dropped) is ~half → clearly wrong, not a rounding effect.

Run:  micromamba run -n blocking python -m pytest tests/test_metric_term.py -q
  or: micromamba run -n blocking python tests/test_metric_term.py
"""
import sys
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from pvtend.sh_ops import vortdiv_sh
from pvtend.constants import R_EARTH

U0 = 30.0  # m/s solid-body amplitude
BAND = (20.0, 70.0)  # mid-latitude band for the FD comparison (away from pole/equator)


def _solid_body_grid(nlat=73, nlon=144):
    """Regular lat-lon grid (poles included) with u = U0 cosφ, v = 0."""
    lat = np.linspace(-90.0, 90.0, nlat)          # S→N ascending (pyspharm regular grid)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    phi = np.deg2rad(lat)
    u = (U0 * np.cos(phi))[:, None] * np.ones((nlat, nlon))
    v = np.zeros((nlat, nlon))
    return lat, lon, phi, u, v


def _analytic_zeta(phi):
    return 2.0 * U0 * np.sin(phi) / R_EARTH        # (nlat,) broadcast over lon


def _flat_and_metric(u, v, lat, lon, phi):
    """Flat curl ∂v/∂x−∂u/∂y (metric dropped) and the metric term u·tanφ/a."""
    a = R_EARTH
    # v = 0 here so ∂v/∂x = 0; ∂u/∂y = (1/a) ∂u/∂φ (central FD on the grid)
    dudphi = np.gradient(u, phi, axis=0)
    dudy = dudphi / a
    zeta_flat = -dudy                              # ∂v/∂x(=0) − ∂u/∂y
    metric = u * np.tan(phi)[:, None] / a          # +u·tanφ/a
    return zeta_flat, metric


def _band_mask(lat):
    return (np.abs(lat) >= BAND[0]) & (np.abs(lat) <= BAND[1])


def test_metric_term_is_present_in_sh_vorticity():
    lat, lon, phi, u, v = _solid_body_grid()
    zeta_an = _analytic_zeta(phi)[:, None] * np.ones_like(u)
    zeta_sh, _ = vortdiv_sh(u, v, lat, lon)
    zeta_flat, metric = _flat_and_metric(u, v, lat, lon, phi)
    zeta_spherical = zeta_flat + metric

    m = _band_mask(lat)
    an = zeta_an[m]
    scale = np.max(np.abs(an))

    # 1) spherical FD (flat + metric) reproduces the analytic vorticity
    assert np.max(np.abs(zeta_spherical[m] - an)) / scale < 1e-2, \
        "flat FD + u·tanφ/a should recover the analytic ζ"

    # 2) the SH operator reproduces the analytic vorticity (metric baked in)
    assert np.max(np.abs(zeta_sh[m] - an)) / scale < 5e-3, \
        "spherical-harmonic ζ should match analytic ζ (metric term included)"

    # 3) the flat FD (metric dropped) is ~half the truth → clearly wrong
    rel_err_flat = np.max(np.abs(zeta_flat[m] - an)) / scale
    assert rel_err_flat > 0.3, \
        "flat FD must deviate substantially (it drops +u·tanφ/a)"

    # for solid-body rotation the flat curl is *exactly* half the analytic ζ
    assert np.allclose(zeta_flat[m], 0.5 * an, rtol=2e-2), \
        "flat FD equals half the analytic ζ for solid-body rotation"


if __name__ == "__main__":
    lat, lon, phi, u, v = _solid_body_grid()
    zeta_an = _analytic_zeta(phi)
    zeta_sh, _ = vortdiv_sh(u, v, lat, lon)
    zeta_flat, metric = _flat_and_metric(u, v, lat, lon, phi)
    m = _band_mask(lat)
    j = np.argmin(np.abs(lat - 45.0))
    a_amp = np.max(np.abs(zeta_an[m]))
    print(f"solid-body rotation u=U0 cosφ (U0={U0} m/s), a={R_EARTH:.0f} m")
    print(f"  at 45°N: analytic ζ = {zeta_an[j]:.3e} /s")
    print(f"           SH ζ       = {zeta_sh[j, 0]:.3e} /s  (metric baked in)")
    print(f"           flat FD ζ  = {zeta_flat[j, 0]:.3e} /s  (metric DROPPED)")
    print(f"           metric u·tanφ/a = {metric[j, 0]:.3e} /s")
    print(f"  flat FD / analytic (band max ratio) = "
          f"{np.max(np.abs(zeta_flat[m]))/a_amp:.3f}  (≈0.5 → half)")
    test_metric_term_is_present_in_sh_vorticity()
    print("PASS: SH vorticity includes the +u·tanφ/a metric term; flat FD does not.")
