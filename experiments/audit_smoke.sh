#!/usr/bin/env bash
# EXPERIMENT: LD_AUDIT vs LD_PRELOAD parity. Runs the basic-block-counter demo (bb_count.sh)
# under BOTH host injectors and diffs the per-wave output, proving the rtld-audit port
# (runtime/host/audit.so) drives the identical end-to-end flow as the LD_PRELOAD build
# (runtime/host/preload.so): fatbin substitution, instrumented-co detection, executable
# augmentation (mailbox + hostcall lib), cross-object reloc resolution at freeze, the
# GPU->CPU hostcall service, and the launch-time per-wave buffer.
#
# The two injectors differ ONLY in plumbing (see runtime/host/hostcall_hooks.h):
#   preload.so : LD_PRELOAD shadow symbols; real fn = dlsym(RTLD_NEXT)
#   audit.so   : la_symbind redirect;       real fn = walk the app lib's .dynsym
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"

make -C "$ROOT/runtime/host" preload.so audit.so >/dev/null

pass=1
for inj in preload audit; do
  echo "########## HC_INJECTOR=$inj ##########"
  HC_INJECTOR=$inj bash "$ROOT/experiments/bb_count.sh" | sed -n '/run under/,$p'
  echo
done

P="$ROOT/experiments/runs/bb_count.preload"
A="$ROOT/experiments/runs/bb_count.audit"
echo "== parity check: per-wave counts (preload vs audit) =="
shopt -s nullglob
for f in "$P"/bbcount_*.txt; do
  b=$(basename "$f")
  if [ -f "$A/$b" ] && diff -q "$f" "$A/$b" >/dev/null; then
    echo "  $b: IDENTICAL"
  else
    echo "  $b: DIFFERS (or missing under audit)"; pass=0
  fi
done
# both runs must have reported a correct kernel result + serviced the hostcall
grep -q PASSED "$P/run.log" || { echo "  preload run did not PASS"; pass=0; }
grep -q PASSED "$A/run.log" || { echo "  audit run did not PASS";   pass=0; }

echo
if [ "$pass" = 1 ]; then echo "PASS: LD_AUDIT matches LD_PRELOAD end to end"; else echo "FAIL: divergence detected"; exit 1; fi
