#!/usr/bin/env bash
# EXPERIMENT: multiple CODE OBJECTS. Kernel kA in the exe (twoco) + kernel kB in a HIP
# shared library (libkb.so) = two fatbins -> two hsa_executable_t. A manifest drives
# both substitutions; the preload augments BOTH executables. Verifies both traced.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

make -C "$MUTATEES" twoco libkb.so >/dev/null
make -C "$ROOT/runtime/device" >/dev/null

inst() {  # $1=binary $2=name $3=kernel $4=idbase
  "$OBJDUMP" --offloading "$1" >/dev/null 2>&1
  cp -f "$(ls -t "$1".0.hipv4*gfx908* | head -1)" "$MUTATEES/$2.co"
  DYNINST_ID_BASE=$4 "$MUTATORS/multikernel_instrument" \
    "$MUTATEES/$2.co" "$MUTATEES/$2.inst.co" "$RUNTIME_LIB" "$3" >/dev/null
  cp -f "$MUTATEES/$2.inst.co" "$MUTATEES/$2.inst.synced.co"
  python3 "$TOOLS/sync_note_from_kd.py" "$MUTATEES/$2.inst.synced.co" >/dev/null
  printf '' > /tmp/empty.host
  "$BUNDLER" --type=o --targets=host-x86_64-unknown-linux-gnu-,$TARGET \
    --input=/tmp/empty.host --input="$MUTATEES/$2.inst.synced.co" --output="$MUTATEES/$2.bundle" 2>/dev/null
}
inst "$MUTATEES/twoco"    co_kA _Z2kAPfPKfS1_i 0        # exe co
inst "$MUTATEES/libkb.so" co_kB _Z2kBPfPKfS1_i 1        # .so co

RUN="$ROOT/experiments/runs/multi_co"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
cat > manifest <<EOF
$MUTATEES/co_kA.co $MUTATEES/co_kA.inst.synced.co $MUTATEES/co_kA.bundle
$MUTATEES/co_kB.co $MUTATEES/co_kB.inst.synced.co $MUTATEES/co_kB.bundle
EOF
HOSTCALL_MANIFEST="$RUN/manifest" HOSTCALL_LIB="$RUNTIME_LIB" \
  LD_PRELOAD="$PRELOAD" "$MUTATEES/twoco" 2>&1 | grep -iE 'SUBSTITUT|augmented|serviced|PASS|FAIL'
echo "== trace (site 0 = exe-co kA, site 1 = .so-co kB): =="; sort dyninst_trace.txt | uniq -c
