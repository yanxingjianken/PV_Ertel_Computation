# 03 — Interpolate σ-PV to isobaric / isentropic surfaces

Reads the σ-PV netCDF from step 02 and samples the PV onto **pressure** levels
(`interp_to_pressure`, per-column log-p) and/or **isentropic** θ levels
(`interp_to_isentropic`, per-column in θ; default {300,315,320,330,350} K — the
RWB/blocking "middleworld" set). Both interpolators live in
[`../src/ertel_pv.py`](../src/ertel_pv.py) and are shared with `test00`.

```bash
micromamba run -n blocking python interp_pv_to_levels.py \
    --pv-sigma-nc ../02_compute_pv_hybrid/out/cesm2le_m01_pv_sigma_2010-01-01.nc \
    --output-coords pressure,isentropic
# --theta-levels 315,330,350   --pressure-levels 500,300,250,200   to override
```

**Output** (`out/`): `..._pv_pressure_<date>.nc`, `..._pv_isentropic_<date>.nc`
+ plots in `out/plots/`.

Note: this **samples** the σ-level PV at θ / p — the relative vorticity ζ was
evaluated on σ surfaces (Option A), validated against ERA5 native PV-on-θ in
`test00`. The exact isentropic route (ζ recomputed on θ) is the upgrade path if
needed near jets.
