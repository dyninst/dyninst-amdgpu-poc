#!/usr/bin/env bash
# hecbench_analyze.sh — SURVEY workstream A3.
# For every gfx908 .co in the corpus, extract SIMT-control-flow metrics into metrics.csv:
#   - raw idiom counts from disasm (saveexec / execz / execnz / v_cndmask / v_cmp / insns)
#   - recovered structure from simtcfg (CONDITION / MASKFLIP=else / RECONVERGE / SimtIf groupings)
#   - recovered source-level conditions from `structure` (if / loops / max nesting depth)
# Robust: per-tool timeouts, tolerates failures, parallel. Reusable stashed binaries in BIN.
#
#   usage: hecbench_analyze.sh [CO_DIR]     (default /home/wuxx1279/hecbench-survey/co)
#   env: JOBS (parallel, default 8)
set -u
POC=${POC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # repo root (overridable)
ROCM=${ROCM:-/opt/rocm-7.0.2}
OBJDUMP="$ROCM/lib/llvm/bin/llvm-objdump"
READELF="$ROCM/lib/llvm/bin/llvm-readelf"
BIN=${BIN:-$POC/analysis/cfrecovery/build}                     # simtcfg/structure binaries (override if stashed elsewhere)
CODIR=${1:-/home/wuxx1279/hecbench-survey/co}
JOBS=${JOBS:-8}
export LD_LIBRARY_PATH="$POC/build/dyninst/lib64:${LD_LIBRARY_PATH:-}"
export BIN OBJDUMP READELF
CSV="$CODIR/metrics.csv"

analyze_one() {
  local co="$1" b; b=$(basename "$co" .co)
  local D S T mcpu
  mcpu=$("$READELF" -h "$co" 2>/dev/null | grep -oiE 'gfx[0-9a-f]+' | head -1); mcpu=${mcpu:-gfx908}
  D=$(timeout 60 "$OBJDUMP" -d --arch=amdgcn --mcpu="$mcpu" "$co" 2>/dev/null)
  local insns se ez enz cm vc
  insns=$(printf '%s' "$D" | grep -cE '^[0-9a-f ]+:\s')
  se=$(printf '%s' "$D" | grep -cE 'saveexec')
  ez=$(printf '%s' "$D" | grep -cE 's_cbranch_execz')
  enz=$(printf '%s' "$D" | grep -cE 's_cbranch_execnz')
  cm=$(printf '%s' "$D" | grep -cE 'v_cndmask')
  vc=$(printf '%s' "$D" | grep -cE 'v_cmp')
  S=$(timeout 60 "$BIN/simtcfg" "$co" 2>/dev/null)
  local cond mf rc si
  cond=$(printf '%s' "$S" | grep -cE '  CONDITION ')
  mf=$(printf '%s' "$S" | grep -cE '  MASKFLIP ')
  rc=$(printf '%s' "$S" | grep -cE '  RECONVERGE ')
  si=$(printf '%s' "$S" | grep -cE 'SimtIf #')
  T=$(timeout 90 "$BIN/structure" "$co" 2>/dev/null)
  local k rif lp mn
  k=$(printf '%s' "$T" | grep -cE '^==========')
  rif=$(printf '%s' "$T" | grep -cE '^[[:space:]]*if ')
  lp=$(printf '%s' "$T" | grep -c 'loop hdr')
  mn=$(printf '%s' "$T" | grep -E '^[[:space:]]*if ' | \
        awk '{n=0; while(substr($0,n+1,1)==" ")n++; d=int(n/2); if(d>m)m=d} END{print m+0}')
  echo "$b,$k,$insns,$se,$ez,$enz,$cm,$vc,$cond,$mf,$rc,$si,$rif,$lp,$mn"
}
export -f analyze_one

echo "bench,kernels,insns,saveexec,execz,execnz,cndmask,vcmp,cond,maskflip,reconv,simtif,rec_if,loops,max_nest" > "$CSV"
ls "$CODIR"/*.co 2>/dev/null | xargs -P"$JOBS" -I{} bash -c 'analyze_one "$@"' _ {} >> "$CSV"
echo "wrote $CSV : $(( $(wc -l < "$CSV") - 1 )) code objects analyzed"
