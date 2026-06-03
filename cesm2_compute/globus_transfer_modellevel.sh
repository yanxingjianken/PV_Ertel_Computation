#!/usr/bin/env bash
# ==============================================================================
# Transfer NATIVE MODEL-LEVEL (hybrid sigma-pressure) CESM2-LE daily data from
# GLADE (NCAR GDEX d651056) to dolma via Globus, for Ertel PV computation.
#
# Unlike the AWS LENS2 zarr (which drops the hybrid coefficients), these CAM
# history files embed hyam/hybm/hyai/hybi/P0 — required to reconstruct the TRUE
# pressure p = hyam*P0 + hybm*PS(x,y) of each model level.
#
# Variables: U,V,T (cam.h6, 3-D on 32 hybrid levels) + PS (cam.h1, 2-D).
#
# Usage:
#   module load apps/globusconnectpersonal/3.2.2
#   globus login           # NCAR OIDC (Kerberos + Duo), once per session
#   bash globus_transfer_modellevel.sh <MEMBER> <DECADE>
#     MEMBER : 1..10   (first 10 LENS2 members)
#     DECADE : 20102014 (default) | 19851989 | 19901999 | 20002009
#
# GCP only exposes $HOME, so files land in home staging; the compute script
# reads them there and writes PV output to data2, then deletes the raw files.
# ==============================================================================
set -euo pipefail

MEMBER="${1:?Usage: $0 <MEMBER> <DECADE>}"
DECADE="${2:-20102014}"

export TMPDIR="/net/flood/data2/users/x_yan/tmp"
mkdir -p "$TMPDIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/transfer_modellevel_m${MEMBER}_d${DECADE}_$(date +%F_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] === Native model-level transfer  M${MEMBER}  D${DECADE} ==="

module load apps/globusconnectpersonal/3.2.2 2>/dev/null || true

GCP_DOLMA="ae8d0ae3-4c75-11f0-88e6-02fa2a4031ab"
NCAR_GLADE="d33b3614-6d04-11e5-ba46-22000b92c6ec"
REMOTE_BASE="/glade/campaign/collections/gdex/data/d651056/CESM2-LE/atm/proc/tseries/day_1"

# GCP endpoint is rooted at $HOME → use a home-relative path for the destination.
STAGE_REL="cmip6/d651056/modellevel_m${MEMBER}_d${DECADE}"
STAGE_ABS="/net/flood/home/x_yan/${STAGE_REL}"
mkdir -p "$STAGE_ABS"

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
INIT="${INIT_MAP[$MEMBER]:?Unknown member: ${MEMBER}}"
DEC_RANGE="${DECADE_RANGE[$DECADE]:?Unknown decade: ${DECADE}}"
MEM_STR="$(printf '%03d' "$MEMBER")"
CASE="b.e21.BHISTcmip6.f09_g17.LE2-${INIT}.${MEM_STR}"

if ! globus whoami > /dev/null 2>&1; then
  echo "FATAL: not authenticated. Run:  globus login"
  exit 1
fi

# (var, cam-stream) — U/V/T are model-level h6; PS is 2-D h1.
declare -A VAR_STREAM=( [U]=h6 [V]=h6 [T]=h6 [PS]=h1 )

submit_and_wait() {
  local var="$1" stream="$2"
  local fname="${CASE}.cam.${stream}.${var}.${DEC_RANGE}.nc"
  local dest_abs="${STAGE_ABS}/${fname}"
  if [[ -f "$dest_abs" ]]; then
    echo "[$(date -Is)]   ${var}: already staged ($(du -h "$dest_abs" | cut -f1))"
    return 0
  fi
  local src="${NCAR_GLADE}:${REMOTE_BASE}/${var}/${fname}"
  local dst="${GCP_DOLMA}:${STAGE_REL}/${fname}"
  local out task status
  out=$(globus transfer "$src" "$dst" --label "PV-ml-${var}-m${MEMBER}-d${DECADE}" 2>&1)
  task=$(echo "$out" | grep -oP 'Task ID:\s*\K\S+' || true)
  if [[ -z "$task" ]]; then
    echo "[$(date -Is)]   ${var}: ERROR submitting -> $out"; return 1
  fi
  echo "[$(date -Is)]   ${var}: task ${task} submitted; waiting ..."
  # large files (~20 GB) — poll up to ~2 h
  for ((i=0; i<480; i++)); do
    status=$(globus task show "$task" 2>/dev/null | awk -F': *' '/^Status/{print $2}')
    case "$status" in
      SUCCEEDED) echo "[$(date -Is)]   ${var}: SUCCEEDED ($(du -h "$dest_abs" 2>/dev/null | cut -f1))"; return 0 ;;
      FAILED)    echo "[$(date -Is)]   ${var}: FAILED"; return 1 ;;
    esac
    sleep 15
  done
  echo "[$(date -Is)]   ${var}: TIMEOUT"; return 1
}

rc=0
for var in T PS U V; do          # smallest first → fail fast / quick header check
  submit_and_wait "$var" "${VAR_STREAM[$var]}" || rc=1
done

# GCP can only write to $HOME; move the raw files onto data2 (large, 70 TB free)
# so home staging stays clear.
DATA2_DIR="${SCRIPT_DIR}/globus_data/m${MEMBER}_d${DECADE}"
if [[ $rc -eq 0 ]]; then
  mkdir -p "$DATA2_DIR"
  echo "[$(date -Is)] Moving raw files  home -> ${DATA2_DIR} ..."
  for f in "${STAGE_ABS}"/*.nc; do
    [[ -f "$f" ]] || continue
    mv -v "$f" "${DATA2_DIR}/"
  done
  rmdir "$STAGE_ABS" 2>/dev/null || true
  touch "${DATA2_DIR}/.transfer_complete"
  echo "[$(date -Is)] All vars on data2 at: ${DATA2_DIR}"
  echo "DATA_DIR=${DATA2_DIR}"
else
  echo "[$(date -Is)] One or more transfers failed (rc=$rc) — files left in ${STAGE_ABS}."
  echo "STAGE_DIR=${STAGE_ABS}"
fi
exit $rc
