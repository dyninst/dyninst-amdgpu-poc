#!/usr/bin/env bash
# EXPERIMENT: MULTIPLE per-wave variables at distinct offsets (the multi-var / per-arg
# offset-add path). Declares two per-wave vars (v0 @0, v1 @8) and inserts
# pw_mark2(v0.address(), v1.address()) @exit, which writes 0xAAAA to v0 and 0xBBBB to v1.
# Runs under the LAUNCHER (native HSA dispatch => rocgdb-attachable, and it dumps each
# wave's slice). PASS = word[0]==0xAAAA (43690) AND word[2]==0xBBBB (48059): the two vars
# landed at base+0 and base+8, proving emitCall's per-arg VGPR offset-add.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL=_Z9vectoraddPfPKfS1_i
EXE="$MUTATEES/vectoradd"
make -C "$MUTATEES" vectoradd >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null

echo ">> [1] extract app co"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
command cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"

echo ">> [2] instrument pw_mark2(v0@0, v1@8) @exit"
OUT=$("$MUTATORS/multivar_instrument" "$EXE.co" "$EXE.mv.co" "$KERNEL" "$USER_LIB" 2>/dev/null)
echo "$OUT" | grep -E '^multivar'
STRIDE=$(echo "$OUT" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')
echo "   per-wave STRIDE = ${STRIDE} B"

echo ">> [3] sync .note; bake __dyninst_pw_stride"
python3 "$TOOLS/sync_note_from_kd.py" "$EXE.mv.co" >/dev/null
"$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.mv.co"

echo ">> [4] run under launcher (dumps per-wave slice; native dispatch)"
RUN="$ROOT/experiments/runs/multivar"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
timeout 60 env ROCR_VISIBLE_DEVICES=1 "$ROOT/runtime/host/launcher" "$EXE.mv.co" "$USER_LIB" "$KERNEL.kd" \
  > run.log 2>&1 || true
grep -iE "STRIDE|counts\[0..3\]|slots written|PASSED|FAILED|fault|illegal" run.log | head -8

echo "== check: word[0]=43690 (0xAAAA), word[2]=48059 (0xBBBB) =="
if grep -qE "counts\[0..3\] = 43690 [0-9]+ 48059 " run.log; then
  echo "PASS: both per-wave vars landed at their distinct offsets"
else
  echo "FAIL: markers not at expected offsets (see run.log)"
fi
