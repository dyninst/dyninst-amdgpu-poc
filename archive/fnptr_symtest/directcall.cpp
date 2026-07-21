// Does ParseAPI cut off at a DIRECT call too, or only at indirect ones?
#include <hip/hip_runtime.h>

__device__ __noinline__ float dcall(float x) { return x + 1.0f; }

__global__ void kd(float* out, float v) {
    float r = dcall(v);              // single DIRECT call (resolvable target)
    out[threadIdx.x] = r * 2.0f;     // code AFTER the call (store + endpgm)
}
