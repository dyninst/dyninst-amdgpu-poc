#!/usr/bin/env bash
# EXPERIMENT: MULTI-WAVE basic-block counter, one per-wave variable PER block (multi-var
# style). Kernel bbmw has WAVE-id-dependent control flow (wave `wid` runs the loop body
# wid+1 times; only odd waves take an extra block), so the per-wave block counts DIFFER
# across waves. N=512 => 8 waves => 8 distinct bbcount_<wid>.txt. The mutator declares one
# per-wave counter per block and inserts bb_inc_one(ctr[k].address()) — the probe just bumps
# the cell it is handed (no bbid/indexing); Dyninst bakes each block's offset via the
# per-arg add. Proves: multiple waves × multiple blocks × independent per-wave counts.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL='_Z4bbmwPiPKii'
make -C "$MUTATEES" bbmw >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null       # combined.aliased.elf: bb_inc_one/bb_flush_pw
EXE="$MUTATEES/bbmw"

echo ">> [1] extract app co + assert getpc-free"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then
  echo "   ERROR: $KERNEL contains s_getpc; getpc-relocation not yet supported"; exit 1
fi

echo ">> [2] instrument: one per-wave counter per block + bb_flush_pw@exit"
INST=$("$MUTATORS/bb_count_multivar_instrument" "$EXE.co" "$EXE.inst.co" "$KERNEL" "$USER_LIB" 2>/dev/null)
echo "$INST" | grep -E '^bb_count_multivar:'
STRIDE=$(echo "$INST" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')
echo "   per-wave STRIDE = ${STRIDE} B"

echo ">> [3] sync .note; expand_args (+1 kernarg); bake __dyninst_pw_stride"
cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
python3 "$TOOLS/expand_args.py" "$EXE.inst.synced.co" --kernel "$KERNEL" --count 1 | grep -E "kernarg_segment_size"
"$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.inst.synced.co"

echo ">> [4] bundle"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

echo ">> [5] run under preload (N=512 => 8 waves; bbmw has 3 explicit args)"
RUN="$ROOT/experiments/runs/bb_count_multivar"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
PW_NARGS=3 \
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
  ROCR_VISIBLE_DEVICES=1 LD_PRELOAD="$PRELOAD" "$EXE" > run.log 2>&1 || true
grep -iE 'per-wave STRIDE|PASSED|FAILED|serviced|fault|illegal' run.log || true

echo "== per-wave basic-block counts (one row per wave; they should DIFFER) =="
for f in $(ls bbcount_*.txt 2>/dev/null | sort -V); do
  printf "%-14s %s\n" "$f:" "$(tr '\n' ' ' < "$f")"
done
n=$(ls bbcount_*.txt 2>/dev/null | wc -l)
uniq=$(for f in bbcount_*.txt; do tr '\n' ' ' < "$f"; echo; done | sort -u | wc -l)
echo "== $n waves, $uniq distinct count profiles =="
[ "$n" -ge 8 ] && [ "$uniq" -ge 8 ] && echo "PASS: every wave shows a different per-block profile" \
                                     || echo "NOTE: expected 8 waves each distinct"
