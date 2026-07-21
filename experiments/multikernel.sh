#!/usr/bin/env bash
# EXPERIMENT: multiple kernels in ONE code object. Instruments both kadd and kmul of
# twokernels; each logs a distinct id at entry. Verifies both kernels are serviced.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

make -C "$MUTATEES" twokernels >/dev/null
make -C "$ROOT/runtime/device" >/dev/null
EXE="$MUTATEES/twokernels"

# multikernel_instrument is variadic (<in> <out> <lib> <kernel...>), so instrument directly.
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
"$MUTATORS/multikernel_instrument" "$EXE.co" "$EXE.inst.co" "$RUNTIME_LIB" \
  _Z4kaddPfPKfS1_i _Z4kmulPfPKfS1_i >/dev/null
cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

RUN="$ROOT/experiments/runs/multikernel"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
  HOSTCALL_LIB="$RUNTIME_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
  LD_PRELOAD="$PRELOAD" "$EXE" 2>&1 | grep -iE 'serviced|PASS|FAIL'
echo "== trace (site 0 = kadd, site 1 = kmul): =="; sort dyninst_trace.txt | uniq -c
