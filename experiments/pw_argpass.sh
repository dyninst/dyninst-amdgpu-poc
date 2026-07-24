#!/usr/bin/env bash
# EXPERIMENT (step 1, isolation): pass an EXTRA per-wave-buffer pointer to a kernel via
# the kernarg, with NO dyninst instrumentation. Validates purely the arg-passing path:
#   - tools/expand_args.py adds one global_buffer arg descriptor (+grows kernarg_segment_size)
#   - the preload appends g_pw_buf to hipLaunchKernel's args[] (PW_NARGS = orig explicit count)
#   - HIP copies that pointer into the kernarg; gdb confirms kernarg[+280] == g_pw_buf.
# See pw_argpass_gdb.sh for the rocgdb check.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL='_Z6bbdemoPiPKii'
make -C "$MUTATEES" bbdemo >/dev/null
EXE="$MUTATEES/bbdemo"

echo ">> [1] extract app co"
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1 || true
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"

echo ">> [2] expand_args: +1 global_buffer arg (uninstrumented co)"
cp -f "$EXE.co" "$EXE.exp.co"
python3 "$TOOLS/expand_args.py" "$EXE.exp.co" --kernel "$KERNEL" --count 1

echo ">> [3] bundle expanded co"
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.exp.co" --output="$EXE.exp.bundle" 2>/dev/null

echo ">> [4] run under preload (PW_NARGS=3 appends g_pw_buf as the 4th explicit arg)"
RUN="$ROOT/experiments/runs/pw_argpass"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
PW_NARGS=3 \
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.exp.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.exp.bundle" \
  LD_PRELOAD="$PRELOAD" "$EXE" > run.log 2>&1 || true
echo "== preload / app log =="
grep -iE 'per-wave buffer|appended per-wave|PASSED|FAILED|error|fault' run.log || true
echo "(full log: $RUN/run.log)"
