// vectoradd2d — 2D-grid vectoradd mutatee (for the 2D/3D per-wave wid work).
//
// A 2D launch (grid.y>1, block.y>1) exercises the general wavefront-id path: the
// per-wave slice index must flatten (blockIdx.{x,y}, threadIdx.{x,y}, blockDim, gridDim)
// consistently with the host's nwaves = numBlocks * ceil(blockDim/64), or slices collide.
// getpc-free (no device calls / rodata), so it is instrumentable by the current substrate.
//
// Build contract matches the runtime/user lib (gfx908:sramecc+:xnack- + COV6).

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

__global__ void vectoradd2d(float* C, const float* A, const float* B, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) { int i = y * W + x; C[i] = A[i] + B[i]; }
}

int main() {
    HIP_CHECK(hipSetDevice(select_gfx908_device()));

    const int W = 64, H = 64;                     // 4096 elems; block16x16=4 waves, grid4x4 => 64 waves
    const int N = W * H;
    const size_t bytes = (size_t)N * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { hA[i] = (float)i; hB[i] = (float)(2 * i); }

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice));

    dim3 block(16, 16);                            // 256 threads => 4 waves/block
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    hipLaunchKernelGGL(vectoradd2d, grid, block, 0, 0, dC, dA, dB, W, H);
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
