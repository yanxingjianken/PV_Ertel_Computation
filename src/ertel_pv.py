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
"""
import numpy as np

G = 9.80665; OMEGA = 7.29212e-5; R_D = 287.0; C_P = 1004.0
P0 = 100000.0; KAPPA = R_D / C_P; PVU_SCALE = 1.0e6

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
    """Interpolate pressure→sigma. All arrays surface→top (descending p/σ).
    np.interp requires increasing x, so we work in ascending (top→surface)
    and flip back."""
    nlev,nlat,nlon=field_plev.shape; nsig=len(sigma_levels)
    log_plev_asc=np.log(plev_Pa[::-1])
    field_sigma=np.empty((nsig,nlat,nlon),dtype=field_plev.dtype)
    for j in range(nlat):
        for i in range(nlon):
            p_sig=sigma_levels*ps[j,i]
            log_p_asc=np.log(np.maximum(p_sig[::-1],1e-3))
            col_asc=field_plev[::-1,j,i]
            ia=np.interp(log_p_asc,log_plev_asc,col_asc,left=col_asc[0],right=col_asc[-1])
            field_sigma[:,j,i]=ia[::-1]
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


def ertel_pv_sigma(u, v, t, plev, ps, lat, lon, sigma_levels=None, method="simple"):
    """Ertel PV on terrain-following sigma levels.

    Returns both sigma-level PV and ERA5 PV interpolated to sigma for
    direct comparison on the same surfaces.

    Returns
    -------
    pv_sigma : (nsig, nlat, nlon)  PV computed on sigma levels [PVU].
    pv_era5_sigma : (nsig, nlat, nlon)  Original p-level PV interpolated
        to sigma for comparison [PVU] (pass as pv_era5) — NOT computed
        internally; caller must provide via *pv_era5_plev*.
    """
    ndim=u.ndim; nla=lat.shape[0]
    if sigma_levels is None: sigma_levels=plev/P0
    sigma_levels=np.asarray(sigma_levels,dtype=float)
    if sigma_levels[0]<sigma_levels[-1]: sigma_levels=sigma_levels[::-1]
    nsig=len(sigma_levels)

    fc=2.0*OMEGA*np.sin(np.deg2rad(lat)).reshape([1]*(ndim-2)+[nla,1])
    u_s=_interp_to_sigma(u,plev,ps,sigma_levels)
    v_s=_interp_to_sigma(v,plev,ps,sigma_levels)
    t_s=_interp_to_sigma(t,plev,ps,sigma_levels)

    # actual pressure at each sigma level: p(σ) = σ × ps(x,y)
    p_s3d=sigma_levels[:,np.newaxis,np.newaxis]*ps[np.newaxis,:,:]
    theta_s=_compute_theta(t_s,p_s3d)
    zeta_s=_vorticity_sh(u_s,v_s,lat,lon)
    dthdx_s,dthdy_s=_gradient_sh(theta_s,lat,lon)
    dudσ=_vert_deriv(u_s,sigma_levels); dvdσ=_vert_deriv(v_s,sigma_levels)
    dthdσ=_vert_deriv(theta_s,sigma_levels)

    inv_ps_3d=(1.0/np.maximum(ps,1.0))[np.newaxis,:,:]
    pv_stretch=(fc+zeta_s)*dthdσ; pv_s=-G*inv_ps_3d*pv_stretch
    if method=="full": pv_s=-G*inv_ps_3d*(pv_stretch+dvdσ*dthdx_s-dudσ*dthdy_s)
    return pv_s*PVU_SCALE, p_s3d
