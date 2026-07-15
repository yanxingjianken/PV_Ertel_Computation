#!/usr/bin/env bash
# ==============================================================================
# Step 01 (NCAR machines) — SELECTIVE SYMLINK of native model-level CESM2-LE data.
#
# On an NSF NCAR machine (Casper / Derecho) the GDEX d651056 archive is directly
# mounted under /glade, so there is NO need to transfer anything: just symlink the
# few files step 02 needs (U,V,T on 32 hybrid levels + PS) into a local stage dir.
# This is the NCAR-side equivalent of globus_transfer_modellevel.sh (dolma-side).
#
# Files linked (one member, one decade):
#   U,V,T : cam.h6  (3-D, 32 hybrid levels, hyam/hybm/hyai/hybi/P0 embedded)
#   PS    : cam.h1  (2-D surface pressure)
#
# Usage:
#   bash symlink_ncar.sh <MEMBER> <DECADE> [<DEST_DIR>]
#     MEMBER   : 1..10   (first 10 LENS2 members; extend INIT_MAP for more)
#     DECADE   : 20102014 (default) | 19851989 | 19901999 | 20002009
#     DEST_DIR : where to place the symlinks (default: ./glade_symlinks/m<M>_d<DEC>)
#
# Then run step 02 with  --stage-dir <DEST_DIR>.
# ==============================================================================
set -euo pipefail

MEMBER="${1:?Usage: $0 <MEMBER> <DECADE> [<DEST_DIR>]}"
DECADE="${2:-20102014}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DEST_DIR="${3:-${SCRIPT_DIR}/glade_symlinks/m${MEMBER}_d${DECADE}}"

REMOTE_BASE="/glade/campaign/collections/gdex/data/d651056/CESM2-LE/atm/proc/tseries/day_1"

if [[ ! -d "$REMOTE_BASE" ]]; then
  echo "FATAL: $REMOTE_BASE not found."
  echo "  This script is for NCAR machines (Casper/Derecho) where /glade is mounted."
  echo "  On dolma or elsewhere, use globus_transfer_modellevel.sh instead."
  exit 1
fi

declare -A INIT_MAP=(
  [1]=1001 [2]=1021 [3]=1041 [4]=1061 [5]=1081
  [6]=1101 [7]=1121 [8]=1141 [9]=1161 [10]=1181
)
declare -A DECADE_RANGE=(
  [19851989]="19850101-19891231"
  [19901999]="19900101-19991231"
  [20002009]="20000101-20091231"
  [20102014]="20100101-20141231"
)
INIT="${INIT_MAP[$MEMBER]:?Unknown member: ${MEMBER} (extend INIT_MAP)}"
DEC_RANGE="${DECADE_RANGE[$DECADE]:?Unknown decade: ${DECADE}}"
MEM_STR="$(printf '%03d' "$MEMBER")"
CASE="b.e21.BHISTcmip6.f09_g17.LE2-${INIT}.${MEM_STR}"

# (var, cam-stream) — U/V/T are model-level h6; PS is 2-D h1.
declare -A VAR_STREAM=( [U]=h6 [V]=h6 [T]=h6 [PS]=h1 )

mkdir -p "$DEST_DIR"
echo "=== NCAR symlink  M${MEMBER} (INIT ${INIT})  D${DECADE}  ->  ${DEST_DIR} ==="
rc=0
for var in T PS U V; do
  stream="${VAR_STREAM[$var]}"
  fname="${CASE}.cam.${stream}.${var}.${DEC_RANGE}.nc"
  src="${REMOTE_BASE}/${var}/${fname}"
  dst="${DEST_DIR}/${fname}"
  if [[ ! -f "$src" ]]; then
    echo "  MISSING: ${src}"; rc=1; continue
  fi
  ln -sfn "$src" "$dst"
  echo "  linked ${var}: ${dst} -> ${src}"
done

if [[ $rc -eq 0 ]]; then
  echo "All 4 vars linked. Next:"
  echo "  micromamba run -n blocking python ../02_compute_pv_hybrid/compute_pv_hybrid.py \\"
  echo "      --stage-dir ${DEST_DIR} --member ${MEMBER} --date 2010-01-01"
else
  echo "One or more source files missing (see above) — check member/decade or the archive layout."
fi
exit $rc
