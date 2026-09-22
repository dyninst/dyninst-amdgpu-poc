#!/usr/bin/env bash
# hecbench_sweep.sh — SURVEY workstream A1-A2.
# Compile every HeCBench *-hip benchmark to gfx908 and extract its device code object,
# building a corpus for the static SIMT-CFG survey. Resilient: per-benchmark timeout,
# skip+log failures, resumable (skips benchmarks whose .co already exists), parallel.
#
#   usage: hecbench_sweep.sh [SRC_DIR] [OUT_DIR]
#     SRC_DIR default: /home/wuxx1279/hecbench-survey/HeCBench/src
#     OUT_DIR default: /home/wuxx1279/hecbench-survey/co
#   env: ROCM (default /opt/rocm-7.0.2), TMO (per-bench build timeout s, default 240), JOBS (parallel, default 8)
set -u
ROCM=${ROCM:-/opt/rocm-7.0.2}
HIPCC="$ROCM/bin/hipcc"
OBJDUMP="$ROCM/lib/llvm/bin/llvm-objdump"
SRC=${1:-/home/wuxx1279/hecbench-survey/HeCBench/src}
OUT=${2:-/home/wuxx1279/hecbench-survey/co}
ARCH=${ARCH:-gfx908:sramecc+:xnack-}
TMO=${TMO:-240}
JOBS=${JOBS:-8}
mkdir -p "$OUT"
STATUS="$OUT/status.csv"
[ -f "$STATUS" ] || echo "bench,result,co" > "$STATUS"

build_one() {
  local dir="$1" name; name=$(basename "$dir")
  local co="$OUT/$name.co"
  [ -f "$co" ] && { echo "$name,cached,$co"; return; }
  timeout "$TMO" make -C "$dir" CC="$HIPCC" HIP_ARCH="$ARCH" \
     EXTRA_CFLAGS="--offload-arch=$ARCH -mcode-object-version=6" -j2 \
     >/dev/null 2>&1
  local rc=$? exe="$dir/main"
  if [ $rc -ne 0 ] || [ ! -x "$exe" ]; then echo "$name,build_fail,"; make -C "$dir" clean >/dev/null 2>&1; return; fi
  "$OBJDUMP" --offloading "$exe" >/dev/null 2>&1
  local src_co; src_co=$(ls -t "$exe".0.*amdgcn* 2>/dev/null | head -1)   # device co, any gfx9 arch
  if [ -z "$src_co" ]; then echo "$name,no_co,"; make -C "$dir" clean >/dev/null 2>&1; rm -f "$exe".*hipv4* 2>/dev/null; return; fi
  cp -f "$src_co" "$co"
  echo "$name,ok,$co"
  make -C "$dir" clean >/dev/null 2>&1; rm -f "$exe".*hipv4* 2>/dev/null   # reclaim space, keep source
}
export -f build_one; export OUT HIPCC OBJDUMP ARCH TMO

ls -d "$SRC"/*-hip 2>/dev/null | \
  xargs -P"$JOBS" -I{} bash -c 'build_one "$@"' _ {} >> "$STATUS"

echo "=== sweep done ==="
awk -F, 'NR>1{c[$2]++} END{for(k in c) print "  "k": "c[k]}' "$STATUS"
echo "  co corpus: $(ls "$OUT"/*.co 2>/dev/null | wc -l) objects in $OUT"
