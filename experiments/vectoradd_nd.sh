#!/usr/bin/env bash
# EXPERIMENT: 2D / 3D per-wave instrumentation — validates the general global-wavefront-id.
#
# For a 2D and a 3D vectoradd, instruments pw_open@entry / pw_flush@exit (one file per wave)
# and checks that the per-wave slice map is a correct BIJECTION over [0, nwaves): the run
# must produce exactly nwaves files wave_0.txt .. wave_{nwaves-1}.txt with no gaps and no
# collisions. That is only true if the emitter's per-wave slice-base wid (emit-amdgpu.C entry
# prologue, the 2D/3D block-linear form) and the probe's global_wavefront_id() agree with the
# host's nwaves = numBlocks * ceil(blockDim/64). A wrong wid shows up as a missing/duplicate
# file (collision) or a fault (out-of-range slice). getpc-free kernels.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

make -C "$MUTATEES" vectoradd2d vectoradd3d >/dev/null
make -C "$ROOT/instrumentation/user_lib" >/dev/null           # pw_open/pw_flush + gpu_fopen

# run_case <exe> <kernel> <nargs> <nwaves>
run_case() {
  local exe="$1" kernel="$2" nargs="$3" nwaves="$4"
  local EXE="$MUTATEES/$exe"
  echo "########## $exe  (kernel $kernel, $nwaves waves) ##########"

  "$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1
  cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
  if "$OBJDUMP" -d "$EXE.co" | grep -q s_getpc; then
    echo "   ERROR: $kernel contains s_getpc; getpc-relocation not yet supported"; return 1
  fi

  local INST; INST=$("$MUTATORS/preload_perwave_instrument" "$EXE.co" "$EXE.inst.co" "$kernel" "$USER_LIB" 2>/dev/null)
  echo "$INST" | grep -E 'wrote|pw_stride'
  local STRIDE; STRIDE=$(echo "$INST" | grep -oE 'pw_stride=[0-9]+' | grep -oE '[0-9]+')

  cp -f "$EXE.inst.co" "$EXE.inst.synced.co"
  python3 "$TOOLS/sync_note_from_kd.py" "$EXE.inst.synced.co" >/dev/null
  python3 "$TOOLS/expand_args.py" "$EXE.inst.synced.co" --kernel "$kernel" --count 1 >/dev/null
  "$ROCM/lib/llvm/bin/llvm-objcopy" --add-symbol "__dyninst_pw_stride=${STRIDE},global" "$EXE.inst.synced.co"

  printf '' > /tmp/empty.host
  "$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
    --input=/tmp/empty.host --input="$EXE.inst.synced.co" --output="$EXE.bundle" 2>/dev/null

  local RUN="$ROOT/experiments/runs/vectoradd_nd.$exe"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
  env PW_NARGS="$nargs" \
    HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.inst.synced.co" \
    HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.bundle" \
    ROCR_VISIBLE_DEVICES=1 LD_PRELOAD="$PRELOAD" "$EXE" > run.log 2>&1 || true
  grep -iE 'PASSED|FAILED|serviced|froze|fault|illegal' run.log || true

  # BIJECTION check: exactly nwaves files, wids exactly {0..nwaves-1}.
  local got; got=$(ls wave_*.txt 2>/dev/null | wc -l)
  local ids; ids=$(ls wave_*.txt 2>/dev/null | sed -E 's/.*wave_([0-9]+)\.txt/\1/' | sort -n | tr '\n' ' ')
  local want; want=$(seq 0 $((nwaves-1)) | tr '\n' ' ')
  echo "   per-wave files: $got (expect $nwaves)"
  if [ "$got" = "$nwaves" ] && [ "$ids" = "$want" ] && grep -q PASSED run.log; then
    echo "   PASS: wids form the exact bijection {0..$((nwaves-1))} and kernel result is correct"
    return 0
  fi
  echo "   FAIL: wid set = [$ids]"; return 1
}

rc=0
run_case vectoradd2d _Z11vectoradd2dPfPKfS1_ii  5 64 || rc=1
echo
run_case vectoradd3d _Z11vectoradd3dPfPKfS1_iii 6 32 || rc=1
echo
[ "$rc" = 0 ] && echo "ALL PASS: 2D + 3D per-wave wid validated" || { echo "SOME FAILED"; exit 1; }
