#include "hip/hip_runtime.h"

// Leaf callee: reads an implicit arg (threadIdx) + an explicit arg.
extern "C" __device__ __noinline__ __attribute__((used))
int bar(int x) {
    return x * 3 + (int)threadIdx.x;
}

// NON-LEAF callee: calls bar twice (must preserve state across calls => real frame),
// reads another implicit arg (blockIdx).
extern "C" __device__ __noinline__ __attribute__((used))
int foo(int a, int b) {
    int s = bar(a) + bar(b);
    return s + (int)blockIdx.x;
}

// Kernel: passes args, keeps a value live across the call, uses the return value.
__global__ void k(int* out, int n) {
    int t = (int)threadIdx.x;
    int r = foo(t, n);      // t is live across this call (used below) => caller-save
    out[t] = r + t;
}
