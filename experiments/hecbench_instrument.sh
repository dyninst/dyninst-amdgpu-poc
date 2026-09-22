#!/usr/bin/env bash
# hecbench_instrument.sh — SURVEY workstream A5: instrument-and-run a REAL getpc-free HeCBench
# kernel on the MI100. Generalizes simt_if.sh (hardwired to bbdemo) to an arbitrary benchmark:
# rebuild exe -> extract co -> simt_if_instrument (seen[k]++ per SIMT divergence point) -> sync
# note -> expand kernarg (+1 per-wave buffer) -> bundle -> run the benchmark's OWN exe under the
# preload injector (which substitutes the instrumented co by byte-match). Reports correctness +
# the per-SIMT-condition seen counts the SIMT-aware instrumentation produces.
#
#   usage: hecbench_instrument.sh <bench-dir> <kernel-mangled> [run-args...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
BDIR="$1"; KERNEL="$2"; shift 2; RUNARGS=("$@")
NAME=$(basename "$BDIR"); ARCH=${ARCH:-gfx908:sramecc+:xnack-}   # override + set TARGET in env for gfx942

echo ">> [1] build $NAME exe (gfx908:sramecc+:xnack- COV6)"
make -C "$BDIR" clean >/dev/null 2>&1
make -C "$BDIR" CC="$ROCM/bin/hipcc" HIP_ARCH="$ARCH" \
   EXTRA_CFLAGS="--offload-arch=$ARCH -mcode-object-version=6" -j4 >/dev/null 2>&1
EXE="$BDIR/main"; [ -x "$EXE" ] || { echo "   build FAILED"; exit 1; }

echo ">> [2] extract app co + assert getpc-free"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
APP=$(ls -t "$EXE".0.*amdgcn* 2>/dev/null | head -1)   # device co, any gfx9 arch
cp -f "$APP" "$EXE.co"
if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then echo "   getpc present; not supported"; exit 1; fi

echo ">> [3] instrument: seen[k]++ at each SIMT-IF condition"
OUT=$("$MUTATORS/simt_if_instrument" "$EXE.co" "$EXE.inst.co" "$KERNEL" "$USER_LIB" 2>&1) || { echo "$OUT"; exit 1; }
echo "$OUT" | grep -E '^simt_if:|condition block @' | head
STRIDE=$(echo "$OUT" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')
[ -n "${STRIDE:-}" ] || { echo "   no stride / no conditions"; exit 1; }

echo ">> [4] sync note + expand kernarg (+1 per-wave buf) + bake stride=$STRIDE"
cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
python3 "$TOOLS/expand_args.py" "$EXE.inst.synced.co" --kernel "$KERNEL" --count 1 | grep -iE "kernarg_segment_size" || echo "   (expand_args: check global_buffer arg)"
"$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.inst.synced.co"

echo ">> [5] bundle"
EMPTY_HOST=$(mktemp); : > "$EMPTY_HOST"
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input="$EMPTY_HOST" --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null
rm -f "$EMPTY_HOST"

echo ">> [6] run instrumented on MI100 (ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES) : ./main ${RUNARGS[*]}"
RUN="$ROOT/experiments/runs/hec_$NAME"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
timeout 150 env HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
    HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
    ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" $(hc_inject) "$EXE" "${RUNARGS[@]}" > run.log 2>&1
RC=$?
echo "   exit=$RC  ($([ $RC -eq 124 ] && echo TIMEOUT/hang || echo completed))"
echo "   -- run.log signals --"
grep -iE 'PASS|FAIL|error|fault|illegal|abort|serviced|froze|verif|correct|mismatch|success' run.log | head -6 || true
tail -3 run.log | sed 's/^/     /'
echo "   == per-SIMT-condition seen counts (seen[k]=#waves reaching SimtIf #k) =="
shopt -s nullglob
for f in bbcount_*.txt; do echo "   -- $f --"; sed 's/^/     /' "$f"; done
