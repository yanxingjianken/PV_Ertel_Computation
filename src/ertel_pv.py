"""
MPAS-like Ertel Potential Vorticity computation on pressure-level data.

Mirrors the discretization choices in MPAS-Atmosphere's
``mpas_pv_diagnostics.F`` (subroutine ``calc_epv``, lines 1221-1316):

* Horizontal derivatives: centred finite differences on a regular lat-lon grid
  (analogous to MPAS's finite-volume flux divergence on the Voronoi mesh).
* Vertical derivatives: centred differences in the interior, one-sided at the
  top and bottom boundaries (exactly mirroring MPAS's ``calc_vertDeriv_center``
  and ``calc_vertDeriv_one``).
* Full 3-term isobaric Ertel PV formula (not the simplified (f+ζ)∂θ/∂p only):

      PV = -g * [ (∂v/∂p)(∂θ/∂x) - (∂u/∂p)(∂θ/∂y) + (f + ζ)(∂θ/∂p) ] * 1e6

  where ζ = ∂v/∂x - ∂u/∂y, f = 2Ω sin(φ), θ = T (p₀/p)^(R_d/c_p).

References
----------
* MPAS Fortran source: ``mpas_toolchain/mpas/src/core_atmosphere/diagnostics/mpas_pv_diagnostics.F``
* Holton, J. R. & Hakim, G. J. (2013). *An Introduction to Dynamic Meteorology*, 5th ed.
* Bluestein, H. B. (1992). *Synoptic-Dynamic Meteorology in Midlatitudes*, Vol. I.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G = 9.80665            # gravitational acceleration [m s⁻²]
OMEGA = 7.29212e-5     # Earth's angular velocity [rad s⁻¹]
R_D = 287.0            # dry-air gas constant [J kg⁻¹ K⁻¹]
C_P = 1004.0           # specific heat at constant pressure [J kg⁻¹ K⁻¹]
P0 = 100000.0          # reference pressure [Pa]
KAPPA = R_D / C_P      # ≈ 0.2857
PVU_SCALE = 1.0e6      # SI → PVU conversion (1 PVU = 10⁻⁶ K m² kg⁻¹ s⁻¹)


def _compute_theta(t, p):
    """Potential temperature θ = T (p₀/p)^κ.

    Parameters
    ----------
    t : np.ndarray  (..., nlev, nlat, nlon) or (nlev, nlat, nlon)
        Temperature [K].
    p : np.ndarray  (nlev,) or broadcastable
        Pressure at each level [Pa] (ascending from surface to top, i.e.
        highest pressure first).

    Returns
    -------
    theta : np.ndarray  same shape as t
        Potential temperature [K].
    """
    # Broadcast p to match t's level axis
    p_bc = _broadcast_p(t, p)
    return t * (P0 / p_bc) ** KAPPA


def _broadcast_p(arr, p):
    """Broadcast 1-D pressure array to match arr's level dimension."""
    ndim = arr.ndim
    shape = [1] * ndim
    # Assume the level axis is either axis=-3 (if time present) or axis=0
    if ndim >= 3:
        lev_axis = -3
    else:
        lev_axis = 0
    shape[lev_axis] = len(p)
    return p.reshape(shape)


def _vert_deriv(f, p):
    """Vertical derivative ∂f/∂p, mirroring MPAS centred / one-sided stencils.

    MPAS ``calc_vertDeriv_center`` averages forward and backward one-sided
    differences.  At the boundaries, a single one-sided difference is used
    (``calc_vertDeriv_one``).

    Parameters
    ----------
    f : np.ndarray  (..., nlev, nlat, nlon)
        Field defined on pressure levels.
    p : np.ndarray  (nlev,)
        Pressure [Pa] at each level (ascending, i.e. surface → top).

    Returns
    -------
    dfdp : np.ndarray  same shape as f
        ∂f/∂p [units of f Pa⁻¹].
    """
    ndim = f.ndim
    lev_axis = -3 if ndim >= 3 else 0
    nlev = f.shape[lev_axis]
    dp = np.diff(p)  # (nlev-1,)

    # One-sided differences (forward / backward)
    # df_dp_forward[k] = (f[k+1] - f[k]) / (p[k+1] - p[k])
    df_dp_fwd = (np.take(f, np.arange(1, nlev), axis=lev_axis) -
                 np.take(f, np.arange(0, nlev - 1), axis=lev_axis))
    # Broadcast dp
    shape_dp = [1] * ndim
    shape_dp[lev_axis] = nlev - 1
    df_dp_fwd = df_dp_fwd / dp.reshape(shape_dp)

    # Centred = average of forward and backward
    # backward[k] = (f[k] - f[k-1]) / (p[k] - p[k-1]) ... which is just fwd[k-1] shifted
    # So centred[k] = 0.5 * (fwd[k] + fwd[k-1]) for interior k
    # Top: fwd[0]; Bottom: fwd[-1]
    # We can compute by averaging adjacent forward diffs
    # centred[k] = 0.5*(fwd[k-1] + fwd[k]) for k=1..nlev-2

    # Build centred array
    dfdp = np.empty_like(f)
    # Top boundary (k = nlev-1, lowest pressure): one-sided backward
    # But our p is surface→top, so "top" is the last index.
    # MPAS: if level==1 (surface) use one-sided forward (to level above)
    #       if level==nVertLevels (top) use one-sided backward (to level below)
    # Here surface is k=0 (highest pressure), top is k=nlev-1 (lowest pressure)
    # Surface (k=0): forward difference (f[1]-f[0])/(p[1]-p[0])
    idx_fwd0 = [slice(None)] * ndim
    idx_fwd0[lev_axis] = 0
    dfdp[tuple(idx_fwd0)] = df_dp_fwd[tuple(idx_fwd0)]

    # Top (k=nlev-1): backward difference = last forward diff
    idx_top = [slice(None)] * ndim
    idx_top[lev_axis] = nlev - 1
    idx_fwd_last = [slice(None)] * ndim
    idx_fwd_last[lev_axis] = nlev - 2
    dfdp[tuple(idx_top)] = df_dp_fwd[tuple(idx_fwd_last)]

    # Interior (k=1..nlev-2): centred = average of adjacent forward diffs
    if nlev > 2:
        for k in range(1, nlev - 1):
            idx_k = [slice(None)] * ndim
            idx_k[lev_axis] = k
            idx_km1 = [slice(None)] * ndim
            idx_km1[lev_axis] = k - 1
            idx_kfwd = [slice(None)] * ndim
            idx_kfwd[lev_axis] = k
            dfdp[tuple(idx_k)] = 0.5 * (df_dp_fwd[tuple(idx_km1)] +
                                         df_dp_fwd[tuple(idx_kfwd)])

    return dfdp


def _horiz_deriv(f, lat, lon, axis):
    """Horizontal gradient on a regular lat-lon grid.

    Uses ``np.gradient`` with Earth-radius scaling.

    Parameters
    ----------
    f : np.ndarray  (..., nlev, nlat, nlon)
    lat : np.ndarray  (nlat,)
        Latitude in degrees.
    lon : np.ndarray  (nlon,)
        Longitude in degrees.
    axis : int
        Axis along which to differentiate (-2 for lat / y, -1 for lon / x).

    Returns
    -------
    deriv : np.ndarray  same shape as f
        ∂f/∂y or ∂f/∂x [units of f m⁻¹].
    """
    R_E = 6.371e6  # Earth radius [m]
    ndim = f.ndim

    if axis == -2 or axis == (ndim - 2):  # latitude / y
        dlat_rad = np.deg2rad(np.gradient(lat))
        # shape for broadcasting
        shape = [1] * ndim
        shape[-2] = len(lat)
        dy = R_E * dlat_rad.reshape(shape)
        return np.gradient(f, axis=axis) / dy
    elif axis == -1 or axis == (ndim - 1):  # longitude / x
        dlon_rad = np.deg2rad(np.gradient(lon))
        coslat = np.cos(np.deg2rad(lat))
        # Prevent division by zero at poles: clip cos(lat) to a small minimum
        # corresponding to half a grid spacing from the pole.
        dlat_deg = np.abs(np.gradient(lat)).mean()
        coslat_min = np.cos(np.deg2rad(90.0 - 0.5 * dlat_deg))
        coslat = np.maximum(coslat, coslat_min)
        shape = [1] * ndim
        shape[-1] = len(lon)
        shape_cos = [1] * ndim
        shape_cos[-2] = len(lat)
        dx = R_E * coslat.reshape(shape_cos) * dlon_rad.reshape(shape)
        return np.gradient(f, axis=axis) / dx
    else:
        raise ValueError(f"axis must be -1 (lon/x) or -2 (lat/y), got {axis}")


def ertel_pv_isobaric(u, v, t, plev, lat, lon, method="full"):
    """Compute Ertel Potential Vorticity on pressure levels.

    Parameters
    ----------
    u : np.ndarray  (nlev, nlat, nlon) or (nt, nlev, nlat, nlon)
        Eastward wind component [m s⁻¹].
    v : np.ndarray  same shape as u
        Northward wind component [m s⁻¹].
    t : np.ndarray  same shape as u
        Temperature [K].
    plev : np.ndarray  (nlev,)
        Pressure levels [Pa], ascending (surface → top of atmosphere).
    lat : np.ndarray  (nlat,)
        Latitude [degrees North].
    lon : np.ndarray  (nlon,)
        Longitude [degrees East].
    method : str, optional
        ``"full"`` (default) — all three terms of the isobaric Ertel PV.
        ``"simple"`` — only the (f + ζ) ∂θ/∂p term.

    Returns
    -------
    pv : np.ndarray  same shape as u
        Ertel potential vorticity [PVU] (1 PVU = 10⁻⁶ K m² kg⁻¹ s⁻¹).

    Notes
    -----
    The full formula is::

        PV = -g * [(∂v/∂p)(∂θ/∂x) - (∂u/∂p)(∂θ/∂y) + (f + ζ)(∂θ/∂p)] * 1e6

    where ζ = ∂v/∂x - ∂u/∂y is the isobaric relative vorticity and
    f = 2Ω sin(φ) is the Coriolis parameter.

    Vertical differencing mirrors MPAS ``mpas_pv_diagnostics.F``: centred
    in the interior, one-sided at the top and bottom boundaries.
    """
    ndim = u.ndim
    lev_axis = -3 if ndim >= 3 else 0

    nlev = len(plev)
    nlat_arr = lat.shape[0]
    nlon_arr = lon.shape[0]

    # ---- potential temperature ----
    theta = _compute_theta(t, plev)

    # ---- Coriolis parameter ----
    f_cor = 2.0 * OMEGA * np.sin(np.deg2rad(lat))  # (nlat,)
    f_cor_3d = f_cor.reshape([1] * (ndim - 2) + [nlat_arr, 1])

    # ---- relative vorticity ζ = ∂v/∂x - ∂u/∂y ----
    dvdx = _horiz_deriv(v, lat, lon, axis=-1)
    dudy = _horiz_deriv(u, lat, lon, axis=-2)
    zeta = dvdx - dudy

    # ---- θ gradients ----
    dthdx = _horiz_deriv(theta, lat, lon, axis=-1)
    dthdy = _horiz_deriv(theta, lat, lon, axis=-2)

    # ---- vertical derivatives ----
    dudp = _vert_deriv(u, plev)
    dvdp = _vert_deriv(v, plev)
    dthdp = _vert_deriv(theta, plev)

    # ---- assemble PV ----
    # Term 1 (vortex stretching): (f + ζ) ∂θ/∂p
    pv_stretch = (f_cor_3d + zeta) * dthdp

    pv = -G * pv_stretch

    if method == "full":
        # Term 2 (shear): (∂v/∂p)(∂θ/∂x) - (∂u/∂p)(∂θ/∂y)
        pv_shear = dvdp * dthdx - dudp * dthdy
        pv = -G * (pv_stretch + pv_shear)

    # Convert SI → PVU
    pv = pv * PVU_SCALE

    return pv
