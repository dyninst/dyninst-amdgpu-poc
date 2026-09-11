#!/usr/bin/env bash
# EXPERIMENT: SIMT-IF condition counter. Instruments each SIMT-IF CONDITION block (a block whose head
# is a saveexec — the divergence point, now a real block because the parser splits at every EXEC
# writer, IA_amdgpu::isModifyExecMask). Inserts seen[k]++ at the condition block ENTRY (before the
# saveexec = the proposal's §3.1 seen_count) + bb_flush_pw@exit. This is the payoff of the in-parser
# SIMT block split: instrumenting the divergence point used to crash (generateBranch/blockInstance);
# with parse-built split blocks it's plain block-entry instrumentation. Verifies the kernel still
# computes correctly (PASSED) and prints each SIMT-IF's per-wave seen count.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL='_Z6bbdemoPiPKii'
make -C "$MUTATEES" bbdemo >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null
EXE="$MUTATEES/bbdemo"

echo ">> [1] extract app co + assert getpc-free"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then
  echo "   ERROR: $KERNEL contains s_getpc; getpc-relocation not yet supported"; exit 1
fi

echo ">> [2] instrument: seen[k]++ @ each SIMT-IF condition block entry + bb_flush_pw@exit"
INST_OUT=$("$MUTATORS/simt_if_instrument" "$EXE.co" "$EXE.inst.co" "$KERNEL" "$USER_LIB" 2>/dev/null)
echo "$INST_OUT" | grep -E '^simt_if:|condition block @'
STRIDE=$(echo "$INST_OUT" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')
echo "   per-wave STRIDE = ${STRIDE} B"

echo ">> [3] sync .note to bumped KD; expand_args (+1 kernarg for the per-wave buffer)"
cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
python3 "$TOOLS/expand_args.py" "$EXE.inst.synced.co" --kernel "$KERNEL" --count 1 | grep -E "kernarg_segment_size"

echo ">> [3b] bake __dyninst_pw_stride=${STRIDE} into the co"
"$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.inst.synced.co"

echo ">> [4] bundle"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

echo ">> [5] run under ${HC_INJECTOR:-preload}"
RUN="$ROOT/experiments/runs/simt_if.${HC_INJECTOR:-preload}"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
env \
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
  ROCR_VISIBLE_DEVICES=1 $(hc_inject) "$EXE" > run.log 2>&1 || true
grep -iE 'PASSED|FAILED|serviced|froze|fault|illegal' run.log || true

echo "== per-wave SIMT-IF condition seen counts (seen[k] = times SimtIf #k reached) =="
for f in bbcount_*.txt; do echo "-- $f --"; cat "$f"; done
