// How does AMDGPU emit a tail call vs a normal call?
#include <hip/hip_runtime.h>

extern "C" __device__ __noinline__ float callee(float x) { return x * 2.0f + 1.0f; }

// callee(x) is in TAIL position -> candidate for tail-call optimization.
extern "C" __device__ __noinline__ float tailcaller(float x) { return callee(x); }

// callee(x) result is used -> NOT a tail call (normal call + return).
extern "C" __device__ __noinline__ float nontail(float x) { return callee(x) + 3.0f; }

__global__ void k2(float* out, float v) { out[threadIdx.x] = tailcaller(v) + nontail(v); }
