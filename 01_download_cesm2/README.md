# 01 — Download CESM2-LE native hybrid-level data

Stage the native model-level CAM files step 02 needs: **U, V, T** on 32 hybrid
sigma-pressure levels (`cam.h6`, with `hyam/hybm/hyai/hybi/P0` embedded) + **PS**
(`cam.h1`). Two interchangeable ways depending on the machine:

| Machine | Script | How |
|---------|--------|-----|
| **dolma** (or any off-NCAR host) | `globus_transfer_modellevel.sh <MEMBER> <DECADE>` | Globus pull from GLADE GDEX `d651056` → `globus_data/m<M>_d<DEC>/`. Needs `globus login` first. |
| **NCAR** (Casper/Derecho) | `symlink_ncar.sh <MEMBER> <DECADE> [DEST]` | `/glade` is mounted — just symlinks the 4 files (no transfer) → `glade_symlinks/m<M>_d<DEC>/`. |

`MEMBER` 1–10 (first 10 LENS2 members; extend the `INIT_MAP` for more).
`DECADE` ∈ {19851989, 19901999, 20002009, 20102014}.

```bash
# dolma:
module load apps/globusconnectpersonal/3.2.2 && globus login   # once per session
bash globus_transfer_modellevel.sh 1 20102014

# NCAR:
bash symlink_ncar.sh 1 20102014
```

Output = a stage dir of native hybrid files. Feed it to step 02 via `--stage-dir`.
A 33 MB day-slice sample is kept at `globus_data/sample_m01_2010-01-01/`.

See [`../docs/cesm_hybrid_levels.md`](../docs/cesm_hybrid_levels.md) for the full
data-access reference (endpoints, member→INIT map, hybrid coefficients).
