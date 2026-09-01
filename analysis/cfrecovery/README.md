# cfrecovery — SIMT control-flow / condition recovery (project A)

Recover source-level conditional structure (if/elseif/else, ternary/select, loop + each
condition) from a **stripped** AMD gfx908 code object, by classifying EXEC-mask / predication
idioms and reconstructing a queryable region tree. Design + evidence: `docs/CF_RECOVERY.md`.

Built against the patched Dyninst submodule (parseAPI + dataflowAPI + instructionAPI).

## Files

**The API** (semantic — entryID + operand roles + register identity, no `format()` parsing):
- `classify.C` — `classify(Instruction) → {Open,Close,Else,Select,Call,Compare,None}`. The
  foundation. Handles context-dependent ops (`s_or/s_mov exec` = CLOSE only if it writes EXEC).
- `structure.C` — `namespace CFR`: `Region` + `StructureAnalysis` (region tree via `classify()` +
  post-dominance reconvergence cross-check; `root()`, `regionOf(Block*)`, semantic `recoverCond()`).

- `classify.h` — the `classify()` foundation as a shared header (`cfr::classify` + EXEC-mask
  helpers), so tools don't each carry their own copy. Used by `cfgraph.C`.

**Visualization:**
- `cfgraph.C` — dumps the CFG in Graphviz DOT (in the spirit of Dyninst's CFGraph example), with
  each block *shaped by its EXEC-mask role* from `classify()`: ▲ diverge (`s_and_saveexec`), ▼
  reconverge (`s_or exec`), else arm, `&&` narrow; each divergence header is paired to its
  post-dominator reconvergence by a dashed edge. `usage: ./cfgraph <co> [func] > out.dot`.
- `make-contact-sheet.sh` — runs `cfgraph` over the whole `~/predicated_control_flows` corpus and
  tiles the rendered PNGs into `out/index.html` (a per-benchmark contact sheet). Generated `*.png`/
  `*.dot` are gitignored; rerun the script to rebuild them.

**Throwaway oracles** (spikes kept to cross-check the API; superseded by the two above):
- `slice_smoke.C` — proves the DataflowAPI `Slicer` reaches the `v_cmp` from a mask (substrate OK).
- `depredicate.C` — condition recovery for `v_cndmask` (predication) and `s_and_saveexec` (EXEC).
- `bracket.C` — the nesting stack idea (string-based prototype; `structure.C` is the real version).

Each file has a `main()` test harness taking `<code-object> [func-substr]`.

## Build

```sh
cmake -S . -B build -DDyninst_DIR=<repo>/build/dyninst/lib64/cmake/Dyninst -Wno-dev
cmake --build build -j4
export LD_LIBRARY_PATH=<repo>/build/dyninst/lib64:$LD_LIBRARY_PATH
./build/structure ~/predicated_control_flows/classic_if_else/code-objects/1-*gfx908 vectoradd
./build/classify  ~/predicated_control_flows/if_elseif_else/code-objects/1-*gfx908 call_fn
```

## Status (2026-08-03)

Verified on the `~/predicated_control_flows` labeled corpus: simple `if` / `if-else` recover as
clean balanced trees with correct conditions + reconvergence; `if_elseif_else` recovers the right
conditions/calls/nesting but flags the outer region `[UNBALANCED]` (the compiler's elseif
backward-branch chain — post-dominance returns the exit). Open: elseif-chain balancing, mask-algebra
recursion (`&&`/`||`), `Condition` AST rooted in the flattened thread ID, loop-vs-if, Slicer-backed
cross-block reaching-defs. See `docs/CF_RECOVERY.md` §4 progress log.
