# Shared environment for the helper + experiment scripts. Source this.
# Override ROCM / DYNINST_PREFIX in the environment if your install differs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM="${ROCM:-/opt/rocm-7.0.2}"
DYNINST_PREFIX="${DYNINST_PREFIX:-/home/wuxx1279/bin/dynamd}"

OBJDUMP="$ROCM/lib/llvm/bin/llvm-objdump"
BUNDLER="$ROCM/lib/llvm/bin/clang-offload-bundler"
TARGET='hipv4-amdgcn-amd-amdhsa--gfx908:sramecc+:xnack-'

RUNTIME_LIB="$ROOT/runtime/device/hostcall_lib.aliased.elf"   # hc_* runtime
USER_LIB="$ROOT/instrumentation/user_lib/combined.aliased.elf" # user probes + runtime
MUTATORS="$ROOT/instrumentation/mutators/build"                # built mutator binaries
PRELOAD="$ROOT/runtime/host/preload.so"
LAUNCHER="$ROOT/runtime/host/launcher"
TOOLS="$ROOT/tools"
MUTATEES="$ROOT/mutatees/vectoradd"
KERNEL_DEFAULT='_Z9vectoraddPfPKfS1_i'

export DYNINSTAPI_RT_LIB="$DYNINST_PREFIX/lib64/libdyninstAPI_RT.so"
# ROCr for the host runtime; dyninst libs for the mutators
export LD_LIBRARY_PATH="$ROCM/lib:$DYNINST_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-1}"       # gfx908 MI100 (device 0 is gfx900)
