// vaddcall3d — 3D-grid vectoradd through a NON-INLINED __device__ call (getpc idiom).
//
// Like vaddcall2d but with a 3D launch (wgidX+wgidY+wgidZ enabled): pushes the
// wavefront-offset SGPR out past three work-group-id SGPRs, and exercises the 3D
// block-linear wid math in the non-leaf scratch entry prologue. getpc idiom present
// (fadd not inlined); non-leaf (has a device call).

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

__global__ void vaddcall3d(float* C, const float* A, const float* B, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < W && y < H && z < D) { int i = (z * H + y) * W + x; C[i] = fadd(A[i], B[i]); }  // getpc -> swappc
}

int main() {
    HIP_CHECK(hipSetDevice(select_gfx908_device()));

    const int W = 16, H = 16, D = 8;
    const int N = W * H * D;
    const size_t bytes = (size_t)N * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { hA[i] = (float)i; hB[i] = (float)(2 * i); }

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice));

    dim3 block(8, 8, 4);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);
    hipLaunchKernelGGL(vaddcall3d, grid, block, 0, 0, dC, dA, dB, W, H, D);
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
