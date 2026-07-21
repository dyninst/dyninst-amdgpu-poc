// Test: do device functions always get symbols, and can a function be reachable
// only via a function pointer?  Four cases with different reachability/linkage.
#include <hip/hip_runtime.h>

typedef float (*fn_t)(float);

// (1) internal linkage, address-taken, reached ONLY through a runtime-selected
//     function pointer (never called directly).
static __device__ __noinline__ float indirect_only(float x) { return x * 3.0f + 1.0f; }

// (2) directly called by the kernel.
__device__ __noinline__ float directly_called(float x) { return x - 2.0f; }

// (3) internal linkage, address-taken but never actually invoked.
static __device__ __noinline__ float addr_taken_unused(float x) { return x + 9.0f; }

// (4) internal linkage, never referenced at all (expect: dead-code eliminated).
static __device__ __noinline__ float never_used(float x) { return x / 7.0f; }

// A device-global table of function pointers (produces relocations to the targets).
__device__ fn_t g_table[2] = { indirect_only, addr_taken_unused };

__global__ void k(float* out, int sel, float v) {
    fn_t f = g_table[sel & 1];   // runtime-selected -> indirect s_swappc
    float r = f(v);
    r += directly_called(v);      // direct call
    out[threadIdx.x] = r;
}
