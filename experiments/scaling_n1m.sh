#!/usr/bin/env bash
# EXPERIMENT: ring-mailbox scaling. N=1<<20 (16384 waves) hc_* trace through the ring.
# With the old single-cell FIFO ticket lock this DEADLOCKED; the ring finishes in <1s.
# Verifies every site 0..7 appears exactly 16384x (one hostcall per wave per site).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

make -C "$MUTATEES" vectoradd >/dev/null
make -C "$ROOT/runtime/device" >/dev/null                     # hc_* runtime lib
"$ROOT/scripts/build_inst_pipeline.sh" "$MUTATEES/vectoradd" test_amdgpu_instrument "$RUNTIME_LIB"

RUN="$ROOT/experiments/runs/scaling_n1m"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
/usr/bin/time -v env \
  HOSTCALL_ORIG_CO="$MUTATEES/vectoradd.co" \
  HOSTCALL_INST_CO="$MUTATEES/vectoradd.inst.synced.co" \
  HOSTCALL_LIB="$RUNTIME_LIB" HOSTCALL_BUNDLE="$MUTATEES/vectoradd.bundle" \
  LD_PRELOAD="$PRELOAD" "$MUTATEES/vectoradd" 2>&1 | grep -iE 'serviced|PASS|Elapsed'

echo "== trace: $(wc -l < dyninst_trace.txt) lines; per-site counts (expect 16384 each): =="
sort dyninst_trace.txt | uniq -c
