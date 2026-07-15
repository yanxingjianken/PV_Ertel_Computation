# test01 — pressure-slice vs native-hybrid PV (accuracy gate)

Optional CESM validation: is it good enough to compute PV from the cheap
pre-computed **pressure-level slices** (`cam.h1` U/V/T at a handful of levels)
instead of the full **native hybrid** column (`cam.h6`, 32 levels)?

`compare_plev_vs_hybrid_pv.py` computes Ertel PV both ways for the same
member/day(s), interpolates each to 850/500/250 hPa, and reports per-level
correlation + NaN-mask agreement. **Gate**: adopt the cheaper plev path only if
`r ≥ 0.95` at all levels and the below-ground NaN masks agree.

```bash
micromamba run -n blocking python compare_plev_vs_hybrid_pv.py \
    --data-dir <dir with cam.h6 U/V/T + cam.h1 U####/V####/T####/PS> \
    --member 1 --dates 2010-01-15,2010-07-15
```

> ⚠️ Known caveat (see repo CHANGELOG): the script assumes U/V/T pressure slices at
> `{10,50,100,200,500,700,850,1000}` hPa, but catalog pages list only
> `{10,200,500,700,850}` for U. Verify the real `cam.h1` level set on GLADE before
> running (tracked separately).

Not part of the 01→02→03 reproduction flow — the production pipeline always uses
native hybrid input (step 02).
