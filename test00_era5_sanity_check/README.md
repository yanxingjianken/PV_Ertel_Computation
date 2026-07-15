# test00 — ERA5 sanity check

Ground-truth validation of the PV core against ERA5's own native `pv` field
(2025-01-08 00Z), on three coordinates with the full 3-term formula:

| Cross-check | vs ERA5 | corr |
|-------------|---------|------|
| **σ** (11 levels) | PV interpolated to σ | 0.89 / 0.92 / 0.97 (σ=0.85/0.50/0.25) |
| **isentropic** (σ→θ) | PV interpolated to θ | 0.93–0.97 (θ=300–350 K) |
| **isobaric** | PV on p-levels | 0.27 / 0.93 / 0.97 (850/500/250 hPa) |

The 850 hPa isobaric corr of 0.27 vs σ's 0.89 shows the below-ground extrapolation
that corrupts isobaric PV over terrain — the reason the pipeline computes on σ.

```bash
# one-time download (needs a CDS API key):
micromamba run -n blocking python download_era5.py
# validate + regenerate all figures in plots/:
micromamba run -n blocking python validate_era5.py
```

Uses the same core (`../src/ertel_pv.py`) as the CESM pipeline (steps 01–03), so a
pass here certifies the shared math before it is applied to CESM2 data.
