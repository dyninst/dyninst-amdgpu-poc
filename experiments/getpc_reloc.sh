#!/usr/bin/env bash
# EXPERIMENT: getpc-call RELOCATION (the general-kernel frontier), 1D / 2D / 3D non-leaf.
#
# vaddcall{,2d,3d} compute C=A+B through a NON-INLINED __device__ call, so each kernel body
# contains the compiler's getpc+add(+addc)->swappc idiom. Instrumenting relocates the
# blocks, moving each s_getpc; its baked add/addc offset must be corrected so fadd still
# resolves (PCWidget-amdgpu.C PCtoReg/IPPatch + CFWidget-amdgpu.C re-emitting the swappc).
# Because these kernels also make a device call they are NON-LEAF, exercising the Gfx908
# scratch entry prologue (FLAT_SCRATCH = flat_scratch_init + wave_offset). 2D/3D push the
# wavefront-offset SGPR out past wgidY/wgidZ. Unlike every other experiment, this one
# REQUIRES s_getpc to be present. Correctness signal = the kernel still PASSES (C==A+B),
# no fault, and parseAPI resolves the relocated swappc to fadd.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

echo ">> build runtime + mutator"
make -C "$ROOT/runtime/device" >/dev/null
cmake --build "$ROOT/instrumentation/mutators/build" --target getpc_reloc_instrument >/dev/null

# run_case <exe> <kernel-mangled>
run_case() {
  local exe="$1" kernel="$2"
  local EXE="$MUTATEES/$exe"
  echo "########## $exe  (kernel $kernel) ##########"
  make -C "$MUTATEES" "$exe" >/dev/null

  echo ">> [1] extract app co + ASSERT it contains s_getpc"
  "$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
  cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
  local NG; NG=$("$OBJDUMP" -d "$EXE.co" | grep -c s_getpc || true)
  echo "   original co has $NG s_getpc instruction(s)"
  [ "$NG" -gt 0 ] || { echo "   ERROR: no s_getpc in $kernel"; return 1; }

  echo ">> [2] instrument (relocates the kernel); DYNINST_DEBUG_RELOC captures the PCtoReg path"
  local RELOG="$ROOT/experiments/runs/getpc_reloc.$exe.mutate.log"; mkdir -p "$(dirname "$RELOG")"
  DYNINST_DEBUG_RELOC=1 "$MUTATORS/getpc_reloc_instrument" \
      "$EXE.co" "$EXE.inst.co" "$kernel" "$RUNTIME_LIB" >"$RELOG" 2>&1 || { cat "$RELOG"; return 1; }
  grep -E '^getpc_reloc:|wrote' "$RELOG" || true
  echo "   routing-to-PCtoReg=$(grep -c 'routing to PCtoReg' "$RELOG")  IPPatch::apply=$(grep -c 'IPPatch::apply (AMDGPU getpc)' "$RELOG")"

  echo ">> [3] sync .note; expand_args (+1 kernarg for the per-wave spill buffer); bake stride"
  cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
  python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
  local STRIDE; STRIDE=$("$OBJDUMP" -d "$EXE.inst.synced.co" 2>/dev/null | grep -oE 's_mulk_i32 s[0-9]+, 0x[0-9a-f]+' | head -1 | grep -oE '0x[0-9a-f]+')
  STRIDE=$((STRIDE)); [ "$STRIDE" -gt 0 ] || STRIDE=4096
  python3 "$TOOLS/expand_args.py" "$EXE.inst.synced.co" --kernel "$kernel" --count 1 | grep -E "kernarg_segment_size" || true
  "$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.inst.synced.co"

  echo ">> [4] bundle"
  printf '' > /tmp/empty.host
  "$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
    --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

  echo ">> [5] run under ${HC_INJECTOR:-preload}"
  local RUN="$ROOT/experiments/runs/getpc_reloc.$exe"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
  env \
    HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
    HOSTCALL_LIB="$RUNTIME_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
    ROCR_VISIBLE_DEVICES=1 $(hc_inject) "$EXE" > run.log 2>&1 || true
  grep -iE 'PASSED|FAILED|serviced|fault|illegal' run.log | sed 's/^/   /' || true

  if grep -q PASSED run.log && ! grep -qiE 'fault|illegal' run.log; then
    echo "   PASS: $exe getpc-relocation correct (C=A+B after relocation)"; return 0
  fi
  echo "   FAIL: see $RUN/run.log and $RELOG"; return 1
}

rc=0
run_case vaddcall   _Z8vaddcallPfPKfS1_i     || rc=1; echo
run_case vaddcall2d _Z10vaddcall2dPfPKfS1_ii || rc=1; echo
run_case vaddcall3d _Z10vaddcall3dPfPKfS1_iii || rc=1; echo
[ "$rc" = 0 ] && echo "ALL PASS: getpc-call relocation validated on 1D + 2D + 3D non-leaf kernels" \
              || { echo "SOME FAILED"; exit 1; }
