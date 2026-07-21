# dyninst-amdgpu-poc

Binary instrumentation of HIP/AMDGPU device code objects (gfx908 / MI100, ROCm 7.0.2):
a patched **Dyninst** statically rewrites a GPU code object to inject GPU→CPU
**hostcalls** from a separately-compiled device library, and an **`LD_PRELOAD` shim**
loads/links/services it transparently inside a real HIP application — no recompilation
of the target.

## Layout

```
dyninst/            # submodule: patched Dyninst (branch bbiiggppiigg/amdgpu-hostcall-multiwave-scratch)
runtime/            # the hostcall runtime — one ABI, both sides
  hostcalls.h       #   the GPU↔CPU mailbox ABI (bounded MPMC ring); shared device+host
  device/           #   device half: hostcall_lib.cpp (gpu_fopen/fwrite + hc_* + ring acquire)
  host/             #   host half: shared service loop + two front-ends
    hostcall_service.{h,cpp}   #   the ring service loop (shared)
    preload.cpp     #     LD_PRELOAD product: hooks __hipRegisterFatBinary + HSA load/freeze
    launcher.cpp    #     standalone diagnostic harness (dispatches a .co directly, no HIP app)
instrumentation/
  user_lib/         # example user device probes (call the runtime) -> combined.aliased.elf
  mutators/         # Dyninst mutators (minimal / per_wave / multikernel / capture / full)
mutatees/vectoradd/ # test HIP apps: vectoradd (+small/mw), twokernels, twoco(+kb_lib)
tools/              # add_object_aliases.py, sync_note_from_kd.py, decode_kd.py, probe/loadtrace.cpp
scripts/            # reusable helpers: env.sh, build_inst_pipeline.sh
experiments/        # reproducible run recipes (scaling, multi-wave/file, multi-kernel, multi-co)
docs/               # ARCHITECTURE.md, HANDOFF.md, ROADMAP.md
archive/            # earlier experiments, preserved for reference
```

## Build & run

```sh
git submodule update --init dyninst          # fetch the patched Dyninst source

# 1. Build + install the patched Dyninst (large; one-time). Prefix defaults below.
cd dyninst && cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/bin/dynamd && \
  cmake --build build -j && cmake --install build && cd ..

# 2. Build the mutators against it, then the PoC components.
make mutators DYNINST_PREFIX=$HOME/bin/dynamd
make build                                   # runtime + user lib + host + mutatees

# 3. Run an experiment (each is self-contained: instrument -> preload -> verify).
bash experiments/scaling_n1m.sh              # 16384 waves through the ring, <1s
bash experiments/multiwave_multifile.sh      # 16 waves -> 16 per-wave files
bash experiments/multikernel.sh              # two kernels in one code object
bash experiments/multi_co.sh                 # exe co + HIP .so co (two executables)
```

Runs target the gfx908 MI100 via `ROCR_VISIBLE_DEVICES=1` (device 0 is gfx900); override
`ROCM` / `DYNINST_PREFIX` in the environment if your installs differ (see `scripts/env.sh`).

## How it fits together

1. **Offline** (`scripts/build_inst_pipeline.sh`): extract the app's actual code object
   (`llvm-objdump --offloading`), run a Dyninst mutator to insert hostcalls, **sync the
   `.note` msgpack to the bumped kernel descriptor** (`tools/sync_note_from_kd.py` — else
   HIP dispatches with the wrong scratch size and the GPU faults), and wrap the result as
   a fatbin bundle.
2. **Runtime** (`runtime/host/preload.so`, `LD_PRELOAD`ed into the app): substitute the
   app's fatbin at `__hipRegisterFatBinary`, inject the hostcall lib + define `mailbox`
   (and the per-wave buffer `g_pw_base`) into each executable at HSA load, and run a CPU
   service thread that answers hostcalls off the ring. Scales per-kernel and
   per-code-object, and to full GPU occupancy (16384 waves).

See `docs/ARCHITECTURE.md` for the full design and `docs/HANDOFF.md` for hard-won gotchas.
