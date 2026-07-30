// spillcall — a kernel that has BOTH its own scratch frame (a dynamically-indexed
// volatile local array => private_segment > 0) AND a non-inlined __device__ call (getpc
// idiom, non-leaf). The array is live ACROSS the call (written before, read after), so
// instrumentation inserted at the call site must not place its IACR/spill region on top
// of the caller's own frame. Tests the region + original + callee sizing/seating.
//
// Result is A[i]+B[i] (the array contributes a provable 0), so a PASSED/FAILED self-check
// detects any corruption of the caller's scratch by our spill region.

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

__attribute__((noinline)) __device__ float fadd(float x, float y) { return x + y; }

__global__ void spillcall(float* C, const float* A, const float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    volatile float local[128];
    for (int j = 0; j < 128; j++) local[j] = (float)((i + j) & 3);   // volatile writes -> scratch frame
    float t = fadd(A[i % N], B[i % N]);                              // CALL while `local` is live
    float s = 0.f;
    for (int j = 0; j < 128; j++) s += local[(j * 11 + i) % 128] * 0.f;  // read AFTER the call; s == 0
    if (i < N) C[i] = t + s;                                         // == A[i] + B[i]
}

int main() {
    HIP_CHECK(hipSetDevice(select_gfx908_device()));

    const int N = 1 << 16;
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
    hipLaunchKernelGGL(spillcall, dim3(grid), dim3(block), 0, 0, dC, dA, dB, N);
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
