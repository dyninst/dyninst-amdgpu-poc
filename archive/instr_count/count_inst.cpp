// instr_count/count_inst.cpp
// Minimal instrumentation library for the dyninst cross-object-call PoC.
//
// Provides a device function that dyninst can insert a call to at each basic
// block, plus a global counter the host reads back after the kernel.
//
// Build:  ./compile.sh
//         -> count_inst_gfx908.hsaco  (a raw gfx908 code object with a
//            .dyninst.count_inst STT_OBJECT alias added by add_object_aliases.py)
//
// Consumer side (dyninst): insert an argument-less call to count_inst at each
// basic block; the inserted R_AMDGPU_ABS64 must target ".dyninst.count_inst"
// so the HSA loader can bind it at freeze (a plain STT_FUNC won't resolve).
//
// Host side: after freeze, look up "g_inst_count" — it's an STT_OBJECT, so
// hsa_executable_get_symbol_by_name + VARIABLE_ADDRESS returns its address
// directly (no cuid-offset trick needed). Read it once the dispatch completes.

#include "hip/hip_runtime.h"

// Global dynamic basic-block-execution counter. extern "C" + used exports it as
// a plain STT_OBJECT named "g_inst_count" that the host can resolve and read.
extern "C" __device__ __attribute__((used))
unsigned long long g_inst_count = 0;

// Called once per instrumented basic block. Warp-aggregated to keep atomic
// traffic low: a single leader lane adds the number of active lanes (= threads
// that executed this block) instead of every lane issuing its own atomicAdd.
extern "C" __device__ __attribute__((used, noinline))
void count_inst() {
    unsigned long long active = __ballot(1);        // 64-bit exec mask (gfx908)
    if (active == 0ull) return;
    unsigned lane   = __lane_id();
    unsigned leader = __ffsll(active) - 1;          // lowest active lane
    if (lane == leader)
        atomicAdd(&g_inst_count, (unsigned long long)__popcll(active));
}
