#!/usr/bin/env bash
# EXPERIMENT: capture a call's ABI return value into a Dyninst-managed per-wave variable,
# then feed it to a later call. Inserts  hv = pw_openfile()  @entry (the call SITE captures
# the returned fopen handle into hv) and  pw_writeln(hv.value())  at a few sites (passes the
# held handle to the fwrite hostcall). Demonstrates the composable "per-wave var holds a
# return value, used later" model. The per-wave slice is one 64-bit handle (arena-sized,
# self-describing via __dyninst_pw_stride). Verifies captured.txt is written.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL=_Z9vectoraddPfPKfS1_i
EXE="$MUTATEES/vectoradd"
make -C "$MUTATEES" vectoradd >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null

echo ">> [1] extract app co (assert getpc-free)"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then
  echo "   ERROR: $KERNEL contains s_getpc; getpc-relocation not supported"; exit 1
fi

echo ">> [2] instrument hv=pw_openfile()@entry + pw_writeln(hv.value())@sites (arena-sized)"
INST=$("$MUTATORS/capture_return_instrument" "$EXE.co" "$EXE.cap.co" "$KERNEL" "$USER_LIB" 2>/dev/null)
echo "$INST" | grep -E '^wrote|pw_writeln'
STRIDE=$(echo "$INST" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')
echo "   per-wave STRIDE = ${STRIDE} B  (one 64-bit handle/wave)"

echo ">> [3] sync .note; expand_args (+1 kernarg); bake __dyninst_pw_stride"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.cap.co" >/dev/null
python3 "$TOOLS/expand_args.py" "$EXE.cap.co" --kernel "$KERNEL" --count 1 | grep -E "kernarg_segment_size"
"$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.cap.co"

echo ">> [4] bundle"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.cap.co" --output="$EXE.cap.bundle" 2>/dev/null

echo ">> [5] run under preload"
RUN="$ROOT/experiments/runs/capture_return"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.cap.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.cap.bundle" \
  ROCR_VISIBLE_DEVICES=1 LD_PRELOAD="$PRELOAD" "$EXE" > run.log 2>&1 || true
grep -iE "STRIDE|serviced|PASSED|FAILED|fault|illegal" run.log || true

echo "== captured.txt =="
[ -f captured.txt ] && { echo "size=$(wc -c < captured.txt) B"; head -2 captured.txt; } || echo "MISSING"
