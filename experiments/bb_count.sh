#!/usr/bin/env bash
# EXPERIMENT: basic-block execution counter. Instruments a getpc-free kernel (bbdemo) —
# inserts bb_inc(pw.address(), bbid) at EVERY basic block + bb_flush_pw(pw.address(), nbb)
# at exit — so each wave writes its per-block execution counts to bbcount_<wid>.txt. The
# per-wave slice is a Dyninst-managed per-wave variable (BPatch_perWaveVar) delivered via
# the launch-time kernarg PerWaveBuf: the preload allocates the buffer and appends it as
# the extra kernarg. Verifies the kernel still computes correctly (PASSED) and prints the
# counts (loop body counted many times; a never-taken block counted 0). getpc-free only —
# see docs/ROADMAP.md "Current frontier".
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL='_Z6bbdemoPiPKii'
make -C "$MUTATEES" bbdemo >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null       # combined.aliased.elf: bb_inc/bb_flush_pw
EXE="$MUTATEES/bbdemo"

echo ">> [1] extract app co + assert getpc-free"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then
  echo "   ERROR: $KERNEL contains s_getpc (calls/rodata); getpc-relocation not yet supported"; exit 1
fi

echo ">> [2] instrument: bb_inc@every block + bb_flush_pw@exit (kernarg PerWaveBuf)"
"$MUTATORS/bb_count_instrument" "$EXE.co" "$EXE.inst.co" "$KERNEL" "$USER_LIB" 2>&1 | grep -E '^bb_count:'

echo ">> [3] sync .note to bumped KD; expand_args (+1 kernarg for the per-wave buffer)"
cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
python3 "$TOOLS/expand_args.py" "$EXE.inst.synced.co" --kernel "$KERNEL" --count 1 | grep -E "kernarg_segment_size"

echo ">> [4] bundle"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

echo ">> [5] run under preload (kernarg PerWaveBuf; bbdemo has 3 explicit args)"
RUN="$ROOT/experiments/runs/bb_count"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
PW_NARGS=3 \
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
  ROCR_VISIBLE_DEVICES=1 LD_PRELOAD="$PRELOAD" "$EXE" > run.log 2>&1 || true
grep -iE 'PASSED|FAILED|serviced|froze|fault|illegal' run.log || true

echo "== per-wave basic-block execution counts =="
for f in bbcount_*.txt; do echo "-- $f --"; cat "$f"; done
