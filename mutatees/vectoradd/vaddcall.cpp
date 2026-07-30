// vaddcall — a vectoradd whose arithmetic is done through NON-INLINED __device__
// functions, so the compiler emits the getpc+add(+addc)->swappc call idiom in the
// kernel body. This is the mutatee for validating getpc-call RELOCATION: when the
// kernel is instrumented (its blocks relocated), each original s_getpc lands at a new
// address and its baked add/addc offset must be corrected (PCWidget-amdgpu.C). If the
// correction is right, fadd/fscale still resolve and out == A+B (PASSED); if it is
// wrong, the calls jump to the wrong address -> fault or garbage (FAILED).
//
// Build contract matches the rest of the tree (gfx908:sramecc+:xnack- + COV6).

#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) do {                                                  \
    hipError_t _e = (cmd);                                                    \
    if (_e != hipSuccess) {                                                   \
        fprintf(stderr, "HIP error %s (%d) at %s:%d -- %s\n",                 \
                hipGetErrorString(_e), _e, __FILE__, __LINE__, #cmd);         \
        std::abort();                                                         \
    }                                                                         \
} while (0)

static int select_gfx908_device() {
    int n = 0; HIP_CHECK(hipGetDeviceCount(&n));
    for (int i = 0; i < n; i++) {
        hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p, i));
        if (strncmp(p.gcnArchName, "gfx908", 6) == 0) {
            printf("Selecting device %d: %s (%s)\n", i, p.name, p.gcnArchName);
            return i;
        }
    }
    fprintf(stderr, "No gfx908 device found.\n"); std::abort();
}

// Two non-inlined device functions => two distinct getpc->swappc call sites in the
// kernel (also exercises target REUSE if the compiler shares the address register).
__attribute__((noinline)) __device__ float fadd(float x, float y)  { return x + y; }
__attribute__((noinline)) __device__ float fscale(float x)         { return x * 1.0f; } // identity

__global__ void vaddcall(float* C, const float* A, const float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float s = fadd(A[i], B[i]);   // getpc -> swappc (fadd)
        C[i]    = fscale(s);          // getpc -> swappc (fscale); result == A+B
    }
}

int main() {
    HIP_CHECK(hipSetDevice(select_gfx908_device()));

    const int N = 1 << 20;
    const size_t bytes = (size_t)N * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { hA[i] = (float)i; hB[i] = (float)(2 * i); }

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (N + block - 1) / block;
    hipLaunchKernelGGL(vaddcall, dim3(grid), dim3(block), 0, 0, dC, dA, dB, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hC, dC, bytes, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) if (hC[i] != hA[i] + hB[i]) { if (errors < 5)
        fprintf(stderr, "mismatch at %d: %f != %f\n", i, hC[i], hA[i] + hB[i]); errors++; }
    printf(errors ? "FAILED with %d errors\n" : "PASSED (%d elements)\n", errors ? errors : N);

    hipFree(dA); hipFree(dB); hipFree(dC);
    free(hA); free(hB); free(hC);
    return errors ? 1 : 0;
}
