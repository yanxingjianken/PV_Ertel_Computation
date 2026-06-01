# pv_ertel_compute — Ertel PV on Sigma Levels with Spherical-Harmonic Gradients

Compute Ertel Potential Vorticity on terrain-following sigma levels ($\sigma = p/p_s$)
from gridded pressure-level atmospheric data (u, v, T, p_s). Uses spectrally accurate
spherical-harmonic horizontal derivatives and MPAS-style vertical differencing.

**Primary product**: sigma-coordinate PV on 11 essential levels (sfc → lower stratosphere).
Isobaric PV available as a backward-compatible reference.

## Formula

$$\text{PV}_\sigma = -\frac{g}{p_s}\left[ \frac{\partial v}{\partial\sigma}\frac{\partial\theta}{\partial x} - \frac{\partial u}{\partial\sigma}\frac{\partial\theta}{\partial y} + (f + \zeta)\frac{\partial\theta}{\partial\sigma} \right] \times 10^6 \ \text{PVU}$$

where $\sigma = p/p_s(x,y)$, $\theta = T(p_0/p)^{R_d/c_p}$ ($p_0 = 1000$ hPa constant),
$\zeta$ from spherical-harmonic vorticity, $f = 2\Omega\sin\phi$.

- **Horizontal derivatives**: `pvtend.sh_ops.gradient_sh` / `vortdiv_sh` — spectral accuracy, no polar singularity
- **Vertical derivatives**: MPAS-style centred interior, one-sided boundaries — coordinate order auto-detected
- **Sigma grid**: 11 essential levels {1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1}

## Variables Needed

| Variable | ERA5 | CESM2-LENS2 | Units |
|----------|------|-------------|-------|
| Eastward wind | `u` | `U` | m s⁻¹ |
| Northward wind | `v` | `V` | m s⁻¹ |
| Temperature | `t` | `T` | K |
| Surface pressure | `sp` | `PS` | Pa |
| Pressure levels | (coordinate) | `lev` | Pa or hPa |
| Latitude | `latitude` | `lat` | °N |
| Longitude | `longitude` | `lon` | °E |

## Project Structure

```
pv_ertel_compute/
├── src/
│   ├── __init__.py
│   └── ertel_pv.py              # Core PV: ertel_pv_sigma() + ertel_pv_isobaric()
├── era_sanity_check/
│   ├── data/                     # Downloaded ERA5 NetCDF (u,v,t,pv,sp)
│   ├── plots/                    # Sigma validation plots
│   ├── download_era5.py          # CDS API download script
│   └── validate_era5.py          # Sigma PV vs ERA5 native PV
├── cesm2_compute/
│   ├── data/                     # CESM2 sample (u,v,t,ps)
│   ├── plots/                    # CESM2 sigma PV plots
│   └── compute_cesm2_pv.py       # AWS S3 download + sigma PV
├── handbook/
│   ├── ertel_pv_handbook.tex     # LaTeX math documentation
│   └── ertel_pv_handbook.pdf     # Compiled PDF
├── README.md
├── CHANGELOG.md                  # → /home/x_yan/.github/session_findings/changelogs/
└── plan.md
```

## Workflow

```mermaid
graph TD
    A[ERA5 CDS API] --> B[download_era5.py]
    B --> C[pl.nc + sp.nc]
    C --> D[validate_era5.py]
    D --> E[src/ertel_pv.py]
    E --> F[ertel_pv_sigma]
    D --> G[era5_pv_sigma_comparison.png]
    D --> H[era5_pv_sigma_rms_profile.png]

    I[CESM2 AWS S3] --> J[compute_cesm2_pv.py]
    J --> K[U,V,T,PS zarr stores]
    K --> E
    J --> L[cesm2le_pv_sigma.png]

    style E fill:#f9f,stroke:#333,stroke-width:2px
```

## Usage

### ERA5 Validation

```bash
cd era_sanity_check
micromamba run -n blocking python validate_era5.py
```

### CESM2-LENS2 PV

```bash
cd cesm2_compute
micromamba run -n blocking python compute_cesm2_pv.py
```

### Python API

```python
from src.ertel_pv import ertel_pv_sigma, DEFAULT_SIGMA_LEVELS

# sigma PV on 11 levels (primary)
pv_sigma, p_3d = ertel_pv_sigma(u, v, t, plev_Pa, ps, lat, lon)
# pv_sigma: (11, nlat, nlon) in PVU
# p_3d:     actual pressure at each σ level

# Custom sigma levels
pv, p3d = ertel_pv_sigma(u, v, t, plev_Pa, ps, lat, lon,
                         sigma_levels=[1.0, 0.85, 0.7, 0.5, 0.3, 0.1])

# Isobaric PV (backward-compatible reference)
from src.ertel_pv import ertel_pv_isobaric
pv_iso = ertel_pv_isobaric(u, v, t, plev_Pa, lat, lon)
```

## Validation (ERA5, 2025-01-08 00Z, 11 σ levels)

| σ | Nom hPa | RMSE | Corr |
|---|---------|------|------|
| 0.850 | 861 | 0.30 PVU | 0.92 |
| 0.500 | 507 | 0.37 PVU | 0.94 |
| 0.250 | 253 | 1.12 PVU | 0.97 |

- **Sigma coordinates fix the near-surface problem**: 850 hPa-level correlation improved from 0.28 (isobaric) to 0.92 (sigma)
- **PV range matches ERA5**: [-51, 71] PVU (sigma) vs [-62, 111] PVU (ERA5 native)
- **0% NaN** in computed sigma PV
- **Stratospheric blow-up eliminated** by capping σ ≥ 0.1 (~100 hPa)

## Environment

| Env | Purpose |
|-----|---------|
| `blocking` | PV computation (pyspharm, windspharm, pvtend, xarray, s3fs) |

```bash
micromamba run -n blocking python <script>.py
```

## Key Design Decisions

- **P0 = 1000 hPa** is the thermodynamic reference for $\theta$ (standard definition, constant). **ps(x,y)** is surface pressure used ONLY for $\sigma = p/p_s$ and divisor $1/p_s$ — two distinct quantities.
- **11 sigma levels** span boundary layer → lower stratosphere. Above ~100 hPa ($\sigma < 0.1$), $\partial\theta/\partial\sigma$ explodes, producing unrealistic PV extremes — excluded by design.
- **Spherical harmonics** replace `np.gradient` — no polar clipping, global spectral accuracy.
- **Dry only** — no moisture/virtual-temperature correction (negligible effect, <0.001 PVU at 500 hPa).
- **Sigma-native computation** follows MPAS best-practice: PV on native model (terrain-following) levels, not isobaric. NCL and MetPy have known issues with sharp vertical PV gradients (AlexLojko, [MPAS forum 2024](https://forum.mmm.ucar.edu/threads/potential-vorticity-calculation-in-mpas-a.16001/)).
| `mpas_toolchain` | PV computation, CESM2 AWS access | xarray, numpy, scipy, s3fs, zarr, matplotlib, cartopy |
| `fourcastnetv2` | ERA5 CDS API download | cdsapi |

## Key Design Decisions

1. **Full 3-term formula** mirrors MPAS's complete dot product $\vec{\omega}_a \cdot \nabla\theta / \rho$
2. **Dry potential temperature** — no moisture correction (matches ERA5/CESM2 pressure-level convention)
3. **Finite differences** (not spectral) — mirrors MPAS discretization philosophy
4. **Centered vertical differences** with one-sided boundaries — exact MPAS analogue
5. **Pole-safe** horizontal derivatives — `cos(lat)` clipped to prevent singularity

## References

- MPAS Fortran: `mpas_toolchain/mpas/src/core_atmosphere/diagnostics/mpas_pv_diagnostics.F`
- Holton & Hakim (2013), *An Introduction to Dynamic Meteorology*, 5th ed.
- CESM2-LENS2 on AWS: https://registry.opendata.aws/ncar-cesm2-lens/
