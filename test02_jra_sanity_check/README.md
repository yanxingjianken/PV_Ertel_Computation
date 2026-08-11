# test02 — JRA-3Q: validate the FULL hybrid → PV → {isobaric, isentropic} chain

`test00` validates the PV formula against ERA5, but ERA5 on GLADE
(`d633000`) has **no model-level stream** — only `e5.oper.an.pl` (pressure) and
`.sfc`. So `test00` can only check PV *given* isobaric input; it never touches the
hybrid → pressure conversion, which is the step unique to CESM2 and the easiest
place to be silently wrong.

**JRA-3Q (RDA `d640000`) is the only reanalysis on GLADE that closes the loop**: it
ships native hybrid model levels *and* an official PV product on isentropic
surfaces.

| stream | what | role |
|---|---|---|
| `anl_mdl/` | **100 native hybrid levels**, 480×960 Gaussian; `tmp ugrd vgrd spfh hgt vvel`; `a_hybrid_level` / `b_hybrid_level` (+ half-level) embedded | **input** |
| `anl_surf/` | `pres` — surface pressure | **input** |
| `anl_isentrop/` | **`pvort`** — PV on θ surfaces (also `pres mntsf ugrd vgrd hgt spfh bvf2`) | **truth** |
| `anl_p/` | pressure-level fields | isobaric cross-check |

Path: `/glade/campaign/collections/rda/data/d640000/<stream>/<YYYYMM>/`

## The point: JRA and CAM hybrid conventions are DIFFERENT

Verified numerically against the file's own `p_hybrid_level_ps_1000mb` column
(ps = 1000 hPa), max error 0.19 Pa (rounding):

| | JRA-3Q | CESM2 / CAM |
|---|---|---|
| formula | **`p = a + b·ps`** | **`p = hyam·P0 + hybm·ps`** |
| A units | **Pa** (`a_hybrid_level`, `units = "Pa"`) | **dimensionless** (`hyam`, normalised by `P0 = 100000 Pa`) |
| level order | **surface → top** (`b[0] = 0.9990`, `b[-1] = 0`) | **top → surface** (`hybm[0] = 0`, `hybm[-1] = 0.9926`) |
| n levels | 100 | 32 |
| grid | 480×960 **Gaussian** | 192×288 regular lat-lon (f09) |

Both differences are load-bearing:

* multiplying JRA's `a` by `P0` gives pressures ~10⁹ Pa off — it fails loudly;
* **the reversed level order fails silently**, because the interpolators only
  require monotonicity, not a particular direction. A flipped profile still
  interpolates, just to the wrong answer.

That is precisely why this test uses a dataset with a *different* convention. On
CESM2 alone there is no second source to disagree with, so a convention error is
invisible; against JRA's own `pvort` it shows up immediately.

## Grid caveat — spherical harmonics on a Gaussian grid

`anl_mdl` exists **only** on the 480×960 Gaussian grid (there is no `anl_mdl125`;
the `*125` 1.25° regular variants exist for `anl_p`, `anl_isentrop`, `anl_surf`
but not for model levels).

`pvtend.sh_ops._get_spharmt` currently hardcodes

```python
sx = Spharmt(nlon, nlat, rsphere=..., gridtype="regular", legfunc="stored")
```

Feeding Gaussian latitudes to `gridtype="regular"` does not raise — it returns
wrong derivatives. Either:

* **(preferred)** thread a `gridtype` argument through `sh_ops`, auto-detecting
  from the latitude spacing (regular ⇔ equally spaced) and defaulting to the
  current behaviour, or
* regrid JRA to regular first — rejected: it inserts an interpolation between the
  input and the truth and blunts exactly what the test is meant to measure.

## Comparisons

1. **isentropic** — our `hybrid → σ → Ertel PV → θ` vs `anl_isentrop/pvort`, on the
   θ surfaces JRA publishes. This is the end-to-end number.
2. **isobaric** — our PV on pressure vs PV recomputed from `anl_p`; isolates the
   hybrid→pressure step from the PV formula.
3. **pressure reconstruction** — our `p(hybrid)` vs `anl_isentrop/pres` (the
   pressure of each θ surface). A pure geometry check with no PV involved, so a
   failure here localises the error to the vertical coordinate alone.

Comparison 3 is the cheap one and should be run first: if `p(θ)` disagrees, every
downstream number is meaningless and the cause is the hybrid convention.

## Status

Design only. Not yet implemented.
