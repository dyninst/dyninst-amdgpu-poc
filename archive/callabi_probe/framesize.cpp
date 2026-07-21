#include "hip/hip_runtime.h"
extern "C" __device__ __noinline__ __attribute__((used))
int leaf_small(int x){ return x*3 + (int)threadIdx.x; }            // leaf: no frame
extern "C" __device__ __noinline__ __attribute__((used))
int frame_small(int x){ return leaf_small(x) + leaf_small(x+1); }  // non-leaf: retaddr frame
extern "C" __device__ __noinline__ __attribute__((used))
int frame_big(int x){ volatile int a[64]; for(int i=0;i<64;i++)a[i]=x+i;
                      int s=0; for(int i=0;i<64;i++)s+=a[i]*leaf_small(i); return s; }  // big frame
extern "C" __device__ __noinline__ __attribute__((used))
int d3(int x){ volatile int a[8]; for(int i=0;i<8;i++)a[i]=x+i; return a[x&7]+(int)threadIdx.x; }
extern "C" __device__ __noinline__ __attribute__((used))
int d2(int x){ volatile int a[8]; for(int i=0;i<8;i++)a[i]=x+i; return a[x&7]+d3(x); }
extern "C" __device__ __noinline__ __attribute__((used))
int d1(int x){ volatile int a[8]; for(int i=0;i<8;i++)a[i]=x+i; return a[x&7]+d2(x); }
extern "C" __device__ __noinline__ __attribute__((used))
int rec(int x){ if(x<=0)return 0; volatile int a[8]; for(int i=0;i<8;i++)a[i]=x+i; return a[x&7]+rec(x-1); }
__global__ void k_leaf (int* o,int n){ o[threadIdx.x]=leaf_small(n); }
__global__ void k_small(int* o,int n){ o[threadIdx.x]=frame_small(n); }
__global__ void k_big  (int* o,int n){ o[threadIdx.x]=frame_big(n); }
__global__ void k_deep (int* o,int n){ o[threadIdx.x]=d1(n); }
__global__ void k_rec  (int* o,int n){ o[threadIdx.x]=rec(n); }
