# CESM2 / CAM Hybrid Sigma-Pressure Levels — Conversion & Globus Data Access

> Canonical reference shared by **Claude** (skill `cesm-hybrid-levels`) and **GitHub
> Copilot** (see `.github/copilot-instructions.md`). Keep this file as the source of
> truth; the Claude skill mirrors it.

## 1. The vertical coordinate (the thing people get wrong)

CAM is on a **hybrid sigma-pressure** grid. `lev` is **NOT** pressure — it is a
*nominal* reference pressure:

    lev_hPa = (hyam + hybm) * P0 / 100      # pressure ONLY IF PS == P0

The **true** pressure of each model level varies with surface pressure:

    p(k, j, i) = hyam(k) * P0 + hybm(k) * PS(j, i)          [Pa]

- `P0 = 100000 Pa`; `hyam` is **dimensionless** (normalized by P0) → multiply by P0.
- `hybm`: pure-sigma weight, `0` at top (pure pressure) → `~0.99` at surface.
- **Midpoints** `hyam`/`hybm` on `lev` (e.g. 32); **interfaces** `hyai`/`hybi` on
  `ilev` (e.g. 33). Midpoints for U/V/T/Q; interfaces for layer mass/edges.
- CAM order is **top → surface**; flip `hyam`/`hybm` with the data if you reorder.

Equivalent canonical refs: geocat-comp `interp_hybrid_to_pressure` /
`pressure_at_hybrid_levels` (`p = a*p0 + b*ps`); NCL `vinth2p`/`pres_hybrid_ccm`
(hya unitless, multiply by p0); CAM User Guide `p = A·P0 + B·Ps`.

**Why it matters:** near-surface levels are ~pure sigma (`hybm≈0.99`); over a 700 hPa
mountain the bottom level's true pressure is ~695 hPa, but `lev` says ~993 hPa — a
~300 hPa error, worst exactly over terrain. Reconstruct `p = hyam*P0 + hybm*PS` before
any pressure/sigma interpolation.

## 2. Converting model levels → pressure or sigma

- **→ isobaric:** build `p3d`, interpolate each column in `log(p)` to target pressures.
- **→ sigma** `σ = p/ps`: per-column target pressure is `σ·ps(j,i)`; interpolate from
  `p3d` in `log(p)`. Model sigma: `σ_model = hyam*P0/PS + hybm`.

In this repo (`src/ertel_pv.py`):

    from src.ertel_pv import ertel_pv_sigma
    pv, p3d_sig = ertel_pv_sigma(u, v, t, None, ps, lat, lon,
                                 hyam=hyam, hybm=hybm, p0=100000.0)
    # ERA5 / already-isobaric: omit hyam/hybm, pass plev (1-D true pressure).

`ertel_pv_sigma` reconstructs `p = hyam*P0 + hybm*PS` via `_interp_to_sigma_3d`
(per-column log-p interp). Without `hyam/hybm` it uses the 1-D `plev` path (correct
only for genuinely isobaric data).

## 3. Getting the coefficients — AWS vs GLADE

- **AWS LENS2 zarr** (`s3://ncar-cesm2-lens`) **drops the coefficients**:
  `atm/static/grid.zarr` has `hyam/hybm` but **all NaN**. Only nominal `lev` is usable.
- **Native CAM history files** (GLADE, GDEX **d651056**) embed
  `hyam/hybm/hyai/hybi/P0`. Use these for any true-pressure / sigma work.

## 4. Pulling GLADE (UCAR) data over Globus

Endpoints (verified):
- **NCAR GLADE** (source): `d33b3614-6d04-11e5-ba46-22000b92c6ec`
- **GCP "dolma"** (dest): `ae8d0ae3-4c75-11f0-88e6-02fa2a4031ab` — GCP exposes **only
  `$HOME`**; move large files to `/net/flood/data2` (~70 TB) after transfer.

Base path (single-variable timeseries; `<freq>` = `day_1`, `hour_6`, `month_1`, …):

    /glade/campaign/collections/gdex/data/d651056/CESM2-LE/atm/proc/tseries/<freq>

- **Daily** (`day_1`): native 3-D (32 hybrid levels, coeffs embedded) `U/ V/ T/ Z3/` as `cam.h6`
  (~18–22 GB/var/decade); surface pressure `PS/` as `cam.h1` (~0.24 GB). ✅ verified on-disk.
- **6-hourly** (`hour_6`): instantaneous 3-D also available — confirm the exact history-tape label
  (`cam.hN`) by listing the d651056 file tree (tape numbering differs from the daily `h6`; don't assume).
- Pressure-level slices alongside: `U500/ U850/ Z500/ …` (already isobaric — cheaper if you only need a
  few standard levels). GDEX `d651056` offers annual/monthly/5-day/daily/6-hourly/3-hourly frequencies.

Naming + member map:

    b.e21.BHIST{cmip6|smbb}.f09_g17.LE2-<INIT>.<MEM_STR>.cam.<tape>.<VAR>.<YYYYMMDD-YYYYMMDD>.nc  (hist)
    b.e21.BSSP370{cmip6|smbb}.f09_g17.LE2-<INIT>.<MEM_STR>.cam.<tape>.<VAR>.<...>.nc                (ssp370)
    decades (hist): 19850101-19891231 19900101-19991231 20000101-20091231 20100101-20141231

How the 100 members are generated (Rodgers et al. 2021; variable names identical across all members —
only <INIT>.<MEM_STR> and the `cmip6`/`smbb` tag change):
- **Macro-only (20 members, MEM_STR=001):** <INIT> = piControl branch year (macro/AMOC state).
    members  1–10 → INIT 1001 1021 1041 1061 1081 1101 1121 1141 1161 1181
    members 91–100 → INIT 1011 1031 1051 1071 1091 1111 1131 1151 1171 1191
- **Micro (80 members):** 4 AMOC-phase states INIT ∈ {1231,1251,1281,1301}, each × 20 members
  (MEM_STR 001…020) differing by an O(1e-14 K) round-off perturbation of atmospheric potential temperature.
- **Biomass burning:** 50 members `cmip6` (original CMIP6 BB) + 50 `smbb` (11-yr smoothed BB), evenly
  spread across init dates; encoded in the case name (`BHISTcmip6` vs `BHISTsmbb`).

Transfer (auth once per session, NCAR OIDC = Kerberos + Duo):

    module load apps/globusconnectpersonal/3.2.2
    globus login
    globus transfer "<GLADE>:<remote>/<file>.nc" "<GCP>:<home-rel>/<file>.nc" --label ...
    globus task show <TASK_ID>          # poll until Status: SUCCEEDED

Turn-key scripts (ordered pipeline — see the top-level README):
- **01** `01_download_cesm2/globus_transfer_modellevel.sh <MEMBER> <DECADE>` — native U/V/T (h6) +
  PS (h1); moves home→`01_download_cesm2/globus_data/m<M>_d<DEC>/`. On NCAR machines use
  `01_download_cesm2/symlink_ncar.sh <MEMBER> <DECADE>` instead (symlinks mounted `/glade`, no transfer).
- **02** `02_compute_pv_hybrid/compute_pv_hybrid.py --stage-dir <dir> --member <M> --date <YMD>`
  — native files → hybrid-correct σ-PV netCDF (`pv_sigma/p_sigma/theta_sigma`) + plot.
- **03** `03_interp_to_levels/interp_pv_to_levels.py --pv-sigma-nc <02 output>` — σ-PV →
  isobaric + isentropic surfaces + plots.

### Keep only a day slice
Globus can't subset a file, so one day still pulls the whole ~20 GB/var decade file.
After computing, extract the date and delete the raw decade files:

    sub = ds.isel(time=slice(idx, idx+1))      # keep time dim, len 1
    sub.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 4} for v in sub.data_vars})

The slice (~10 MB/var) still carries `hyam/hybm/P0`. Worked example:
`01_download_cesm2/globus_data/sample_m01_2010-01-01/`.

## 5. Gotchas
- Verify `lev == (hyam+hybm)*P0/100` to machine precision (confirms `lev` is nominal).
- `TMPDIR=/net/flood/data2/users/x_yan/tmp` (never `/tmp`); env `micromamba run -n blocking`.
- Cap σ ≥ 0.1 (~100 hPa); above that `∂θ/∂σ` blows up.
- ERA5 is already true isobaric — do **not** apply the hybrid formula to it.
