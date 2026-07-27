#!/usr/bin/env bash
# EXPERIMENT: managed PerWaveBuf egress under LD_PRELOAD (getpc-free, moderate grid).
#
# Instruments bbdemo with real_write @exit — a Dyninst-managed per-wave variable whose
# address() is passed to the probe via the kernarg PerWaveBuf. The preload allocates the
# per-wave buffer as MANAGED memory (hipMallocManaged) and appends its pointer as the extra
# kernarg. The probe stages a 2000-byte record (> the 512B by-value cap) into its slice and
# streams it out via the default pass-by-address gpu_fwrite; the host reads the managed
# buffer directly and writes real_<wid>.txt. Verifies the record is intact (2000 bytes).
#
# This is the clean counterpart to the launcher real_write test — it exercises the SAME
# instrumentation through the transparent LD_PRELOAD path. Correctness hinges on the app co
# matching HOSTCALL_ORIG_CO, so we build + extract + run the SAME binary in one pass (an
# earlier ad-hoc attempt failed only because a stale extracted co didn't match the runtime).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL=_Z6bbdemoPiPKii
NBYTES=2000
make -C "$MUTATEES" bbdemo >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null       # combined.aliased.elf (real_write + gpu_fwrite)
EXE="$MUTATEES/bbdemo"

echo ">> [1] extract app co (assert getpc-free) — SAME binary we will run"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then
  echo "   ERROR: $KERNEL contains s_getpc; getpc-relocation not supported"; exit 1
fi

echo ">> [2] instrument real_write(pw.address(), $NBYTES) @exit (arena-sized slice)"
INST=$("$MUTATORS/real_write_instrument" "$EXE.co" "$EXE.pw.co" "$KERNEL" "$USER_LIB" real_write "$NBYTES" 2>/dev/null)
echo "$INST" | grep real_write:
STRIDE=$(echo "$INST" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')
echo "   per-wave STRIDE = ${STRIDE} B  (arena-derived: 64 filename var + ${NBYTES} record var)"

echo ">> [3] sync .note to bumped KD; expand_args (+1 kernarg for the per-wave buffer)"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.pw.co" >/dev/null
python3 "$TOOLS/expand_args.py" "$EXE.pw.co" --kernel "$KERNEL" --count 1 | grep -E "kernarg_segment_size"

echo ">> [3b] bake __dyninst_pw_stride=${STRIDE} into the co (self-describing)"
"$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.pw.co"

echo ">> [4] bundle"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.pw.co" --output="$EXE.pw.bundle" 2>/dev/null

echo ">> [5] run under preload (MANAGED g_arg_buf, PW_NARGS=3 explicit args)"
RUN="$ROOT/experiments/runs/pw_perwave"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
PW_NARGS=3 \
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.pw.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.pw.bundle" \
  ROCR_VISIBLE_DEVICES=1 LD_PRELOAD="$PRELOAD" "$EXE" > run.log 2>&1 || true
grep -iE "SUBSTITUTED|detected instrumented|augmented|per-wave arg buf|serviced|PASSED|FAILED|fault|illegal" run.log || true

echo "== per-wave record files (expect $NBYTES bytes each) =="
shopt -s nullglob
n=0; bad=0
for f in real_*.txt; do
  sz=$(wc -c < "$f"); n=$((n+1)); [ "$sz" -ne "$NBYTES" ] && bad=$((bad+1))
done
echo "files=$n  wrong_size=$bad"
[ "$n" -gt 0 ] && { echo "-- $(ls real_*.txt | head -1) --"; head -c 60 "$(ls real_*.txt | head -1)"; echo; }
