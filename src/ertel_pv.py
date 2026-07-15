"""
MPAS-like Ertel Potential Vorticity with spherical-harmonic gradients.

Mirrors MPAS-Atmosphere vertical discretization (centred interior,
one-sided boundaries) with spectrally accurate spherical-harmonic
horizontal derivatives from ``pvtend.sh_ops``.

Two formulations:
* Isobaric (pressure levels) — PV_p
* Sigma (terrain-following, σ = p/p_s) — PV_σ  [mimics MPAS height levels]

References
----------
* MPAS Fortran: mpas_pv_diagnostics.F
* pvtend SH operators: pvtend/src/pvtend/sh_ops.py
* MPAS PV best-practice: compute on native model (sigma) levels, not isobaric —
  https://forum.mmm.ucar.edu/threads/potential-vorticity-calculation-in-mpas-a.16001/
  (AlexLojko, Mar 2024). We follow this: PV_σ on σ = p/p_s is primary.
"""
import numpy as np

G = 9.80665; OMEGA = 7.29212e-5; R_D = 287.0; C_P = 1004.0
# P0 = 1000 hPa — thermodynamic reference pressure for θ = T·(P0/p)^κ.
# This is a standard physical constant, NOT surface pressure p_s.
# Sigma-coordinate PV uses varying p_s(x,y) for σ = p/p_s and divisor 1/p_s.
P0 = 100000.0; KAPPA = R_D / C_P; PVU_SCALE = 1.0e6

# 11 essential terrain-following sigma levels: σ = p / p_s(x,y)  [sfc→top]
# Boundary layer → lower stratosphere; all stay within ERA5 data range globally.
DEFAULT_SIGMA_LEVELS = np.array([
    1.000, 0.925, 0.850, 0.700, 0.600, 0.500,
    0.400, 0.300, 0.250, 0.200, 0.100,
], dtype=float)

# Default isentropic (θ) output levels [K] — the mid-latitude "middleworld"
# isentropes that intersect the tropopause, standard for RWB / blocking PV maps.
DEFAULT_THETA_LEVELS = np.array([300., 315., 320., 330., 350.], dtype=float)

def _compute_theta(t, p):
    if p.ndim == 1: p_bc = _broadcast_p(t, p)
    else: p_bc = p
    return t * (P0 / p_bc) ** KAPPA

def _broadcast_p(arr, p):
    ndim = arr.ndim; shape = [1] * ndim
    lev_axis = -3 if ndim >= 3 else 0
    shape[lev_axis] = len(p); return p.reshape(shape)

def _vert_deriv(f, coord):
    ndim = f.ndim; lev_axis = -3 if ndim >= 3 else 0; nlev = f.shape[lev_axis]
    f_upper = np.take(f, np.arange(1, nlev), axis=lev_axis)
    f_lower = np.take(f, np.arange(0, nlev - 1), axis=lev_axis)
    dc = np.diff(coord); shape_dc = [1]*ndim; shape_dc[lev_axis] = nlev-1
    df_dc_fwd = (f_upper - f_lower) / dc.reshape(shape_dc)
    dfdc = np.empty_like(f)
    idx_top = [slice(None)]*ndim; idx_top[lev_axis]=nlev-1
    idx_last = [slice(None)]*ndim; idx_last[lev_axis]=nlev-2
    dfdc[tuple(idx_top)] = df_dc_fwd[tuple(idx_last)]
    idx_sfc = [slice(None)]*ndim; idx_sfc[lev_axis]=0
    idx_f0 = [slice(None)]*ndim; idx_f0[lev_axis]=0
    dfdc[tuple(idx_sfc)] = df_dc_fwd[tuple(idx_f0)]
    if nlev > 2:
        for k in range(1, nlev-1):
            idx_k=[slice(None)]*ndim;idx_k[lev_axis]=k
            idx_km1=[slice(None)]*ndim;idx_km1[lev_axis]=k-1
            idx_kf=[slice(None)]*ndim;idx_kf[lev_axis]=k
            dfdc[tuple(idx_k)] = 0.5*(df_dc_fwd[tuple(idx_km1)]+df_dc_fwd[tuple(idx_kf)])
    return dfdc

def _gradient_sh(field, lat, lon):
    from pvtend.sh_ops import gradient_sh
    nlev=field.shape[0]; dfdx=np.empty_like(field); dfdy=np.empty_like(field)
    for k in range(nlev): dfdx[k],dfdy[k]=gradient_sh(field[k],lat,lon)
    return dfdx,dfdy

def _vorticity_sh(u,v,lat,lon):
    from pvtend.sh_ops import vortdiv_sh
    nlev=u.shape[0]; zeta=np.empty_like(u)
    for k in range(nlev): zeta[k],_=vortdiv_sh(u[k],v[k],lat,lon)
    return zeta

def _interp_to_sigma(field_plev, plev_Pa, ps, sigma_levels):
    """Interpolate pressure→sigma. Handles any monotonic plev/sigma ordering.

    Returns field on sigma_levels in the SAME vertical order as sigma_levels.
    Out-of-range (σ·ps outside plev bounds) → NaN.
    """
    nlev, nlat, nlon = field_plev.shape
    nsig = len(sigma_levels)
    plev_Pa = np.asarray(plev_Pa, dtype=float)

    # ── put plev in ascending order for np.interp ──
    if plev_Pa[0] > plev_Pa[-1]:
        plev_asc = plev_Pa[::-1]
        field_asc = field_plev[::-1, :, :]
    else:
        plev_asc = plev_Pa
        field_asc = field_plev

    log_plev_asc = np.log(plev_asc)
    p_min, p_max = plev_asc[0], plev_asc[-1]

    # ── determine sigma_levels orientation ──
    sigma_asc = sigma_levels[0] < sigma_levels[-1]

    field_sigma = np.empty((nsig, nlat, nlon), dtype=field_plev.dtype)
    for j in range(nlat):
        for i in range(nlon):
            p_sig = sigma_levels * ps[j, i]
            # np.interp needs increasing x
            if sigma_asc:
                log_p = np.log(np.maximum(p_sig, 1e-3))
            else:
                log_p = np.log(np.maximum(p_sig[::-1], 1e-3))

            ia = np.interp(log_p, log_plev_asc, field_asc[:, j, i],
                           left=np.nan, right=np.nan)
            if not sigma_asc:
                ia = ia[::-1]
            field_sigma[:, j, i] = ia
    return field_sigma


def _interp_to_sigma_3d(field_plev, p3d, ps, sigma_levels):
    """Interpolate hybrid/3D-pressure source → fixed sigma levels, per column.

    Like :func:`_interp_to_sigma`, but the source pressure is a full 3-D array
    ``p3d`` (nlev, nlat, nlon) instead of a shared 1-D ``plev``. Required for
    terrain-following hybrid coordinates (CAM/CESM2) where the true pressure of
    each model level varies horizontally: p = hyam·P0 + hybm·ps(x,y).

    Returns field on ``sigma_levels`` in the SAME vertical order as
    ``sigma_levels``. Out-of-range (σ·ps outside the column's pressure span)
    → NaN.
    """
    nlev, nlat, nlon = field_plev.shape
    nsig = len(sigma_levels)
    sigma_asc = sigma_levels[0] < sigma_levels[-1]

    field_sigma = np.empty((nsig, nlat, nlon), dtype=field_plev.dtype)
    for j in range(nlat):
        for i in range(nlon):
            p_col = p3d[:, j, i]
            f_col = field_plev[:, j, i]
            # np.interp needs strictly increasing x → ascending pressure
            if p_col[0] > p_col[-1]:
                p_col = p_col[::-1]
                f_col = f_col[::-1]
            log_p_col = np.log(np.maximum(p_col, 1e-3))

            p_sig = sigma_levels * ps[j, i]
            if sigma_asc:
                log_p = np.log(np.maximum(p_sig, 1e-3))
            else:
                log_p = np.log(np.maximum(p_sig[::-1], 1e-3))

            ia = np.interp(log_p, log_p_col, f_col, left=np.nan, right=np.nan)
            if not sigma_asc:
                ia = ia[::-1]
            field_sigma[:, j, i] = ia
    return field_sigma


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════
def ertel_pv_isobaric(u, v, t, plev, lat, lon, method="simple"):
    """Ertel PV on pressure levels (SH gradients)."""
    ndim=u.ndim; nla=lat.shape[0]
    theta=_compute_theta(t,plev)
    fc=2.0*OMEGA*np.sin(np.deg2rad(lat)).reshape([1]*(ndim-2)+[nla,1])
    zeta=_vorticity_sh(u,v,lat,lon)
    dthdx,dthdy=_gradient_sh(theta,lat,lon)
    dudp=_vert_deriv(u,plev); dvdp=_vert_deriv(v,plev); dthdp=_vert_deriv(theta,plev)
    pv_stretch=(fc+zeta)*dthdp; pv=-G*pv_stretch
    if method=="full": pv=-G*(pv_stretch+dvdp*dthdx-dudp*dthdy)
    return pv*PVU_SCALE


def ertel_pv_sigma(u, v, t, plev, ps, lat, lon, sigma_levels=None, method="simple",
                   hyam=None, hybm=None, p0=P0, p3d=None, return_theta=False):
    """Ertel PV on terrain-following sigma levels (σ = p / p_s).

    Two source-coordinate modes:

    * **Isobaric source** (ERA5): pass 1-D ``plev`` — the level coordinate is the
      true pressure. ``hyam/hybm/p3d`` left ``None``.
    * **Hybrid source** (CAM/CESM2): pass ``hyam`` + ``hybm`` (and ``p0``), or a
      precomputed ``p3d``. The true 3-D pressure
      ``p = hyam·p0 + hybm·ps(x,y)`` is reconstructed per column before the σ
      interpolation — ``plev`` is then ignored and may be ``None``. This matches
      the CAM vertical coordinate (equivalently geocat-comp
      ``pressure_at_hybrid_levels`` / ``interp_hybrid_to_pressure``).

    Parameters
    ----------
    u, v, t : (nlev, nlat, nlon)  Wind components [m/s] and temperature [K].
    plev : (nlev,) or None  Pressure levels [Pa], any monotonic order. May be
        ``None`` when ``hyam/hybm`` or ``p3d`` is supplied.
    ps : (nlat, nlon)  Surface pressure [Pa].
    lat, lon : (nlat,), (nlon,)  Latitudes [°S→N], longitudes [°E].
    sigma_levels : (nsig,) optional  σ levels, default 11-level set.
    method : str  "simple" (stretching only) or "full" (all 3 terms).
    hyam, hybm : (nlev,) optional  CAM hybrid A/B midpoint coefficients
        (``hyam`` dimensionless, normalized by ``p0``), aligned with the u/v/t
        level order.
    p0 : float  Reference pressure for the hybrid formula [Pa]. Default 100000.
    p3d : (nlev, nlat, nlon) optional  Precomputed true pressure; overrides
        ``hyam/hybm``.
    return_theta : bool  If True, also return potential temperature θ on the σ
        levels (needed to output isentropic PV via :func:`interp_to_isentropic`).

    Returns
    -------
    pv_sigma : (nsig, nlat, nlon)  Ertel PV on σ levels [PVU].
    p_s3d : (nsig, nlat, nlon)  Actual pressure at each σ level [Pa].
    theta_s : (nsig, nlat, nlon)  Potential temperature on σ levels [K].
        Returned only when ``return_theta=True``.
    """
    ndim = u.ndim; nla = lat.shape[0]
    if sigma_levels is None:
        sigma_levels = DEFAULT_SIGMA_LEVELS.copy()
    sigma_levels = np.asarray(sigma_levels, dtype=float)
    nsig = len(sigma_levels)

    # ── source pressure: hybrid (3-D) vs isobaric (1-D) ──
    if p3d is None and hyam is not None and hybm is not None:
        hyam = np.asarray(hyam, dtype=float)
        hybm = np.asarray(hybm, dtype=float)
        p3d = (hyam[:, np.newaxis, np.newaxis] * p0
               + hybm[:, np.newaxis, np.newaxis] * ps[np.newaxis, :, :])

    fc = 2.0*OMEGA*np.sin(np.deg2rad(lat)).reshape([1]*(ndim-2)+[nla,1])
    if p3d is not None:
        u_s = _interp_to_sigma_3d(u, p3d, ps, sigma_levels)
        v_s = _interp_to_sigma_3d(v, p3d, ps, sigma_levels)
        t_s = _interp_to_sigma_3d(t, p3d, ps, sigma_levels)
    else:
        u_s = _interp_to_sigma(u, plev, ps, sigma_levels)
        v_s = _interp_to_sigma(v, plev, ps, sigma_levels)
        t_s = _interp_to_sigma(t, plev, ps, sigma_levels)

    # Fill NaN at out-of-range σ levels (copy nearest valid level, two-pass)
    for arr in (u_s, v_s, t_s):
        for _ in range(2):  # repeat to handle consecutive NaN
            for k in range(1, nsig):
                fill = np.isnan(arr[k]) & ~np.isnan(arr[k-1])
                if fill.any():
                    arr[k, fill] = arr[k-1, fill]
            for k in range(nsig-2, -1, -1):
                fill = np.isnan(arr[k]) & ~np.isnan(arr[k+1])
                if fill.any():
                    arr[k, fill] = arr[k+1, fill]

    # actual pressure at each sigma level: p(σ) = σ × ps(x,y)
    p_s3d = sigma_levels[:, np.newaxis, np.newaxis] * ps[np.newaxis, :, :]
    theta_s = _compute_theta(t_s, p_s3d)
    zeta_s = _vorticity_sh(u_s, v_s, lat, lon)
    dthdx_s, dthdy_s = _gradient_sh(theta_s, lat, lon)
    dudσ = _vert_deriv(u_s, sigma_levels)
    dvdσ = _vert_deriv(v_s, sigma_levels)
    dthdσ = _vert_deriv(theta_s, sigma_levels)

    inv_ps_3d = (1.0 / np.maximum(ps, 1.0))[np.newaxis, :, :]
    pv_stretch = (fc + zeta_s) * dthdσ
    pv_s = -G * inv_ps_3d * pv_stretch
    if method == "full":
        pv_s = -G * inv_ps_3d * (pv_stretch + dvdσ*dthdx_s - dudσ*dthdy_s)
    pv_s = pv_s * PVU_SCALE
    if return_theta:
        return pv_s, p_s3d, theta_s
    return pv_s, p_s3d


def interp_to_pressure(field_sigma, p_s3d, target_pa):
    """Interpolate a σ-level field onto target pressure levels (per column, log-p).

    Parameters
    ----------
    field_sigma : (nsig, nlat, nlon)  Field on σ levels (e.g. PV from
        :func:`ertel_pv_sigma`).
    p_s3d : (nsig, nlat, nlon)  Actual pressure of each σ level [Pa] (the second
        return value of :func:`ertel_pv_sigma`).
    target_pa : (nout,)  Target pressure levels [Pa].

    Returns
    -------
    (nout, nlat, nlon)  Field on the target pressure levels; out-of-range → NaN.
    """
    field_sigma = np.asarray(field_sigma, dtype=float)
    target_pa = np.asarray(target_pa, dtype=float)
    nsig, nlat, nlon = field_sigma.shape
    out = np.full((len(target_pa), nlat, nlon), np.nan, dtype=float)
    logt = np.log(target_pa)
    for j in range(nlat):
        for i in range(nlon):
            p = p_s3d[:, j, i]; f = field_sigma[:, j, i]
            if np.all(np.isnan(f)):
                continue
            if p[0] > p[-1]:                       # np.interp needs ascending x
                p = p[::-1]; f = f[::-1]
            with np.errstate(invalid="ignore"):
                out[:, j, i] = np.interp(logt, np.log(p), f, left=np.nan, right=np.nan)
    return out


def interp_to_isentropic(field_sigma, theta_s, theta_levels=None):
    """Interpolate a σ-level field onto target isentropic (θ) surfaces, per column.

    θ increases with height in a statically stable atmosphere, so it is a valid
    monotone vertical coordinate. Where θ is locally non-monotone (e.g. a shallow
    superadiabatic surface layer on the coarse σ grid) the profile is made
    monotone non-decreasing (``np.maximum.accumulate``) before interpolation so
    ``np.interp`` stays well defined; this collapses unstable layers and does not
    affect the free-tropospheric / UTLS θ range used for RWB/blocking maps.

    Note (Option A): this SAMPLES the σ-coordinate PV at θ — the relative
    vorticity ζ was evaluated on σ surfaces, not recomputed on the isentrope.
    Validated against ERA5 native PV-on-θ in ``test00_era5_sanity_check``.

    Parameters
    ----------
    field_sigma : (nsig, nlat, nlon)  Field on σ levels (e.g. PV).
    theta_s : (nsig, nlat, nlon)  Potential temperature on the same σ levels [K]
        (from ``ertel_pv_sigma(..., return_theta=True)``).
    theta_levels : (nout,) optional  Target θ levels [K]; default
        :data:`DEFAULT_THETA_LEVELS`.

    Returns
    -------
    (nout, nlat, nlon)  Field on the target θ surfaces; out-of-range → NaN.
    """
    field_sigma = np.asarray(field_sigma, dtype=float)
    theta_s = np.asarray(theta_s, dtype=float)
    if theta_levels is None:
        theta_levels = DEFAULT_THETA_LEVELS.copy()
    theta_levels = np.asarray(theta_levels, dtype=float)
    nsig, nlat, nlon = field_sigma.shape
    out = np.full((len(theta_levels), nlat, nlon), np.nan, dtype=float)
    for j in range(nlat):
        for i in range(nlon):
            th = theta_s[:, j, i]; f = field_sigma[:, j, i]
            good = ~np.isnan(th) & ~np.isnan(f)
            if good.sum() < 2:
                continue
            thc = th[good]; fc = f[good]
            if thc[0] > thc[-1]:                   # ensure surface→top (ascending θ)
                thc = thc[::-1]; fc = fc[::-1]
            thm = np.maximum.accumulate(thc)       # enforce monotone non-decreasing
            # break exact ties left by accumulate so np.interp is single-valued
            for k in range(1, thm.size):
                if thm[k] <= thm[k - 1]:
                    thm[k] = thm[k - 1] + 1e-6
            out[:, j, i] = np.interp(theta_levels, thm, fc, left=np.nan, right=np.nan)
    return out
