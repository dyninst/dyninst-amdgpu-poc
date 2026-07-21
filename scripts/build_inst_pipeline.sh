#!/usr/bin/env bash
# build_inst_pipeline.sh — reusable offline pipeline for ONE code object:
#   extract app co (llvm-objdump --offloading) -> instrument -> sync .note to KD ->
#   wrap as a fatbin bundle. Emits <exe>.{co,inst.co,inst.synced.co,bundle}.
#
# Usage: build_inst_pipeline.sh <exe> [mutator] [lib] [kernel]
#   mutator: name of a built mutator (default test_amdgpu_instrument); must take
#            <in.co> <out.co> <kernel> <lib>  (the common single-kernel signature).
#   lib:     hostcall runtime to link (default the hc_* RUNTIME_LIB).
#   kernel:  target kernel symbol (default vectoradd).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

EXE="$1"
MUTATOR="${2:-test_amdgpu_instrument}"
LIB="${3:-$RUNTIME_LIB}"
KERNEL="${4:-$KERNEL_DEFAULT}"

echo ">> [1] extract app co from $(basename "$EXE")"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
APP="$(ls -t "$EXE".0.hipv4*gfx908* 2>/dev/null | head -1)"
[ -n "$APP" ] || { echo "extract failed (no offload bundle in $EXE)"; exit 1; }
cp -f "$APP" "$EXE.co"

echo ">> [2] instrument with $MUTATOR (kernel $KERNEL, lib $(basename "$LIB"))"
"$MUTATORS/$MUTATOR" "$EXE.co" "$EXE.inst.co" "$KERNEL" "$LIB" >/dev/null

echo ">> [3] sync .note to bumped KD"
cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null

echo ">> [4] wrap as fatbin bundle"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

echo ">> done: $EXE.{co,inst.co,inst.synced.co,bundle}"
