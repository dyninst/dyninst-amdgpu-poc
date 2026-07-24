#!/usr/bin/env bash
# EXPERIMENT: per-wave multi-file write. N=1024 (16 waves); each wave opens its own
# wave_<wid>.txt via the per-wave slice probes (pw_open/pw_flush) and writes a line to it.
# The per-wave slice is a Dyninst-managed per-wave variable (BPatch_perWaveVar) delivered
# via the launch-time kernarg PerWaveBuf: the preload allocates the buffer and appends it
# as the extra kernarg. Verifies 16 distinct files with correct per-wave content.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL=_Z9vectoraddPfPKfS1_i
EXE="$MUTATEES/vectoradd_mw"
make -C "$MUTATEES" vectoradd_mw >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null       # combined.aliased.elf: pw_open/pw_flush

echo ">> [1] extract app co (assert getpc-free)"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then
  echo "   ERROR: $KERNEL contains s_getpc; getpc-relocation not supported"; exit 1
fi

echo ">> [2] instrument pw_open @entry + pw_flush @exit (arena-sized slice)"
INST=$("$MUTATORS/preload_perwave_instrument" "$EXE.co" "$EXE.inst.co" "$KERNEL" "$USER_LIB" 2>/dev/null)
echo "$INST" | grep -E '^wrote'
STRIDE=$(echo "$INST" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')
echo "   per-wave STRIDE = ${STRIDE} B"

echo ">> [3] sync .note to bumped KD; expand_args (+1 kernarg for the per-wave buffer)"
cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
python3 "$TOOLS/expand_args.py" "$EXE.inst.synced.co" --kernel "$KERNEL" --count 1 | grep -E "kernarg_segment_size"

echo ">> [3b] bake __dyninst_pw_stride=${STRIDE} into the co (self-describing)"
"$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.inst.synced.co"

echo ">> [4] bundle"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

echo ">> [5] run under preload (16 waves, PW_NARGS=4 explicit args)"
RUN="$ROOT/experiments/runs/multiwave_multifile"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
PW_NARGS=4 \
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
  ROCR_VISIBLE_DEVICES=1 LD_PRELOAD="$PRELOAD" "$EXE" > run.log 2>&1 || true
grep -iE 'PASSED|FAILED|serviced|froze|fault|illegal' run.log || true

echo "== per-wave files: $(ls wave_*.txt 2>/dev/null | wc -l) (expect 16) =="
cat wave_*.txt 2>/dev/null | sort -t' ' -k2 -n
