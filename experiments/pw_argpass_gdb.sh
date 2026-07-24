#!/usr/bin/env bash
# rocgdb check for step-1 arg passing: break at kernel entry, read the appended arg out
# of the kernarg (base = s[4:5], offset 0x118 = 280), and compare to the buffer pointer
# the preload logged. If they match, HIP copied our appended args[] entry into the kernarg.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

KERNEL='_Z6bbdemoPiPKii'
EXE="$MUTATEES/bbdemo"
ROCGDB=/opt/rocm-7.0.2/bin/rocgdb

# (re)build artifacts the same way pw_argpass.sh does
make -C "$MUTATEES" bbdemo >/dev/null
"$OBJDUMP" --offloading "$EXE" >/dev/null 2>&1 || true
cp -f "$(ls -t "$EXE".0.hipv4*gfx908* | head -1)" "$EXE.co"
cp -f "$EXE.co" "$EXE.exp.co"
python3 "$TOOLS/expand_args.py" "$EXE.exp.co" --kernel "$KERNEL" --count 1 >/dev/null
printf '' > /tmp/empty.host
"$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
  --input=/tmp/empty.host --input="$EXE.exp.co" --output="$EXE.exp.bundle" 2>/dev/null

RUN="$ROOT/experiments/runs/pw_argpass_gdb"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
export PW_NARGS=3 \
  HOSTCALL_ORIG_CO="$EXE.co" HOSTCALL_INST_CO="$EXE.exp.co" \
  HOSTCALL_LIB="$USER_LIB" HOSTCALL_BUNDLE="$EXE.exp.bundle" \
  LD_PRELOAD="$PRELOAD"

echo ">> rocgdb: read kernarg[0x118] at kernel entry"
"$ROCGDB" -batch \
  -ex "set pagination off" \
  -ex "set breakpoint pending on" \
  -ex "break $KERNEL" \
  -ex "run" \
  -ex 'printf "S4=%#x S5=%#x\n", $s4, $s5' \
  -ex 'set $ka = ((unsigned long long)$s5 << 32) + (unsigned int)$s4' \
  -ex 'printf "KERNARG_BASE=%#llx\n", $ka' \
  -ex 'printf "KERNARG[0x118]=%#llx\n", *(unsigned long long *)($ka + 0x118)' \
  -ex "kill" \
  --args "$EXE" 2>&1 | grep -iE "per-wave arg buf|S4=|KERNARG|PASSED|FAILED|error" || true
