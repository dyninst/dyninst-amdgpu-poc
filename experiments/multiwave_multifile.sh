#!/usr/bin/env bash
# EXPERIMENT: per-wave multi-file write. N=1024 (16 waves); each wave opens its own
# wave_<wid>.txt via the global-base per-wave probes (pwg_open/pwg_flush). Verifies 16
# distinct files with correct per-wave content.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

make -C "$MUTATEES" vectoradd_mw >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null           # combined.aliased.elf (user probes)
"$ROOT/scripts/build_inst_pipeline.sh" "$MUTATEES/vectoradd_mw" preload_perwave_instrument "$USER_LIB"

RUN="$ROOT/experiments/runs/multiwave_multifile"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
HOSTCALL_PW_BYTES=$((1<<20)) \
  HOSTCALL_ORIG_CO="$MUTATEES/vectoradd_mw.co" \
  HOSTCALL_INST_CO="$MUTATEES/vectoradd_mw.inst.synced.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$MUTATEES/vectoradd_mw.bundle" \
  LD_PRELOAD="$PRELOAD" "$MUTATEES/vectoradd_mw"

echo "== per-wave files: $(ls wave_*.txt 2>/dev/null | wc -l) (expect 16) =="
cat wave_*.txt 2>/dev/null | sort -t' ' -k2 -n
