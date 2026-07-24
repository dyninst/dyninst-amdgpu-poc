// bbmw.cpp — MULTI-WAVE basic-block counter demo. Control flow depends on the WAVE id
// (uniform per wavefront), so each wave executes the loop a different number of times and
// takes a wave-dependent branch. The per-wave BB counts therefore DIFFER across waves:
// wave `wid` runs the loop body wid+1 times, and only odd waves take the extra block —
// so bbcount_<wid>.txt is distinct for every wave. N=512 => 8 waves (wid 0..7). getpc-free
// (no calls / no PC-relative data; the & (n-1) mask avoids a division libcall).
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void bbmw(int* out, const int* in, int n) {
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = i >> 6;                         // wave id (64 lanes/wave; uniform per wave)
    int acc = 0;
    for (int k = 0; k <= wid; k++)            // loop body executes (wid+1)x per wave
        acc += in[(i + k) & (n - 1)];
    if (wid & 1)                              // extra block taken by ODD waves only
        acc += in[i & (n - 1)] * 3;
    out[i] = acc;
}

static int ref(int i, const int* hi, int n) {
    int wid = i >> 6, acc = 0;
    for (int k = 0; k <= wid; k++) acc += hi[(i + k) & (n - 1)];
    if (wid & 1) acc += hi[i & (n - 1)] * 3;
    return acc;
}

int main() {
    const int N = 512;                        // power of two => 8 waves (wid 0..7)
    size_t sz = N * sizeof(int);
    int *o, *in, ho[N], hi[N];
    for (int i = 0; i < N; i++) hi[i] = i + 1;
    hipMalloc(&o, sz); hipMalloc(&in, sz);
    hipMemcpy(in, hi, sz, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(bbmw, dim3(N / 64), dim3(64), 0, 0, o, in, N);
    hipDeviceSynchronize();
    hipMemcpy(ho, o, sz, hipMemcpyDeviceToHost);
    int ok = 1;
    for (int i = 0; i < N; i++)
        if (ho[i] != ref(i, hi, N)) { ok = 0; printf("mismatch @%d: %d != %d\n", i, ho[i], ref(i, hi, N)); break; }
    printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
