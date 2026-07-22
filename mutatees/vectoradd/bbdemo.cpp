// bbdemo.cpp — a small getpc-free kernel with real control flow (a runtime-bounded loop
// + a conditional), used to demonstrate the basic-block execution counter
// (experiments/bb_count.sh + instrumentation/mutators/bb_count_instrument). getpc-free =
// no calls / no PC-relative data, so mid-block instrumentation works with the current
// relocation support (verify: llvm-objdump -d <co> | grep s_getpc  -> none).
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void bbdemo(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                                 // bounds check -> a never-taken block here
        int acc = 0, m = in[i] & 7;              // 0..7, RUNTIME bound (prevents unroll)
        for (int k = 0; k <= m; k++) {           // loop body: executed multiple times
            if (k & 1) acc += k; else acc -= k;  // conditional inside the loop
        }
        out[i] = acc;
    }
}

static int ref(int v) { int m = v & 7, acc = 0; for (int k = 0; k <= m; k++) { if (k & 1) acc += k; else acc -= k; } return acc; }

int main() {
    const int N = 64; size_t sz = N * sizeof(int);   // one wave -> one bbcount_<wid>.txt
    int *o, *in, ho[N], hi[N];
    for (int i = 0; i < N; i++) hi[i] = i;
    hipMalloc(&o, sz); hipMalloc(&in, sz);
    hipMemcpy(in, hi, sz, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(bbdemo, dim3((N + 63) / 64), dim3(64), 0, 0, o, in, N);
    hipDeviceSynchronize();
    hipMemcpy(ho, o, sz, hipMemcpyDeviceToHost);
    int ok = 1;
    for (int i = 0; i < N; i++) if (ho[i] != ref(hi[i])) { ok = 0; printf("mismatch @%d: %d != %d\n", i, ho[i], ref(hi[i])); break; }
    printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
