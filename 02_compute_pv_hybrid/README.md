# 02 — Compute Ertel PV on the native hybrid (σ) grid

Reads the staged native hybrid files from step 01 and computes the full 3-term
Ertel PV on terrain-following σ levels, reconstructing the true pressure
`p = hyam·P0 + hybm·PS` per column (hybrid-aware `ertel_pv_sigma` in
[`../src/ertel_pv.py`](../src/ertel_pv.py)).

```bash
micromamba run -n blocking python compute_pv_hybrid.py \
    --stage-dir ../01_download_cesm2/globus_data/sample_m01_2010-01-01 \
    --member 1 --date 2010-01-01
```

**Output** (`out/`): `cesm2le_m<MM>_pv_sigma_<date>.nc` carrying `pv_sigma`,
`p_sigma` (actual pressure of each σ level), `theta_sigma` (θ on σ) and `ps` —
the hand-off that step 03 interpolates onto pressure / isentropic surfaces — plus
a σ validation plot in `out/plots/`.

Why compute on σ and interpolate later (not the reverse)? MPAS best-practice:
computing PV on native terrain-following levels avoids smearing the sharp vertical
PV gradients that isobaric-first interpolation introduces. → next: **03**.
