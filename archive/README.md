# archive/

Source of earlier experiments, preserved for reference (build artifacts omitted). These
predate or feed the current hostcall/preload working set:

- `mutatee/`, `mutatee2/` — the original obj2yaml/yaml2obj relocation-edit PoC (redirect a
  device fn-ptr to a separately-compiled `funptr1`), superseded by the Dyninst rewrite path.
- `device_lib/` — the `funptr1` device-only instrumentation stub for that PoC.
- `fnptr_symtest/` — ParseAPI indirect-`s_swappc` truncation reproducers (direct vs indirect
  vs tail-call code objects); characterized the upstream parse bug (fixed by #2326).
- `callabi_probe/` — empirical pinning of the AMDGPU device-fn call ABI (SP/FP/retaddr/stack).
- `scratch_probe/` — hardware-scratch register-spill probes.
- `divergent_mutatee/` — EXEC-divergent control-flow spill validation.
- `alias_test/`, `instr_count/` — misc alias/count experiments.
