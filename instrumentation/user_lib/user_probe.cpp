// user_probe.cpp — a USER-WRITTEN device instrumentation library.
//
// Demonstrates the intended "CPU-like" Dyninst-on-AMDGPU workflow: the user
// writes ordinary device code (the probe/analysis body) that calls into the
// hostcall RUNTIME we provide (hc_write_id — declared here, defined in
// hostcall_lib.cpp). user_probe.cpp is compiled TOGETHER with hostcall_lib.cpp
// into ONE code object, so the user->runtime call is resolved by the device
// linker. Dyninst then instruments a mutatee to call `user_probe`; the user
// never touches the mailbox ABI, the trampoline, or any dyninst-facing detail.
//
// What a probe must satisfy to be a dyninst AMDGPU insertion target:
//   * nullary — dyninst inserts a no-argument call;
//   * __noinline__ + used — so it survives as a real, named, address-taken callee.
// The probe MAY be non-leaf: calling our wrappers makes it non-leaf, and the
// inserted-call ABI sets up the scratch frame + return-address register for it.

#include <cstdint>
#include "hip/hip_runtime.h"

// The runtime we provide. DEFINED in hostcall_lib.cpp, which is compiled into the
// same code object; the device linker binds this call at --genco time.
extern "C" __device__ void hc_write_id(int id);

// USER analysis: count the active lanes at this program point (population count of
// the EXEC mask) and report it through the runtime. __builtin_amdgcn_read_exec is
// valid regardless of ABI setup, so this probe needs no implicit-arg forwarding.
extern "C" __device__ __noinline__ __attribute__((used))
void user_probe() {
    unsigned long long exec = __builtin_amdgcn_read_exec();
    int active_lanes = __builtin_popcountll(exec);   // user's metric
    hc_write_id(active_lanes);                        // our runtime wrapper
}

// PER-WAVE-BUFFER probe: dyninst passes THIS wave's slice of a per-wave buffer as
// the `slice` pointer (a BPatch_perWaveVar). The probe keeps per-wave state there —
// here it just writes a marker to prove the 64-bit pointer arg arrives and points at
// writable, per-wave-distinct memory the host can read back. A real probe would
// accumulate (e.g. a counter/histogram) and periodically flush via gpu_fwrite.
extern "C" __device__ __noinline__ __attribute__((used))
void pw_probe(void* slice) {
    // Elect the first active lane so exactly ONE lane per wave updates the slice
    // (each wave owns its slice, so a plain RMW is race-free — no atomics needed).
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
    // Per-wave slice layout (bytes): [0:7]=file handle  [8]=hits  [12]=lanes  [16..]=name/text.
    if (pos == 0) {
        char* b = (char*)slice;
        *(volatile int*)(b + 8) += 1;                 // hits: # probe sites this wave hit
        *(volatile int*)(b + 12) = __builtin_popcountll(ex);  // active lanes at last hit
    }
}

// Our fopen/fwrite runtime (defined in hostcall_lib.cpp, same code object). Match
// its exact signatures (int64_t handle) so the single-TU reg-usage compile agrees.
extern "C" __device__ int64_t gpu_fopen(const char* filename, const char* mode);
extern "C" __device__ int     gpu_fwrite(int64_t handle, const void* data, int size);      // pass-by-address (default)
extern "C" __device__ int     gpu_fwrite_256(int64_t handle, const char* data, int size);  // by-value, capped 512B

static __device__ int put_str(char* b, int i, const char* s) { while (*s) b[i++] = *s++; return i; }
static __device__ int put_uint(char* b, int i, unsigned v) {
    char t[10]; int n = 0;
    if (!v) t[n++] = '0';
    while (v) { t[n++] = char('0' + v % 10); v /= 10; }
    while (n) b[i++] = t[--n];
    return i;
}

// PER-WAVE OPEN: each wave opens its OWN file "wave_<wid>.txt" and stashes the
// returned handle in its slice, so later writes go to that file. gpu_fopen returns
// the handle only on its elected lane; we do the format+open+store under a single
// elected lane so the stored handle is the real one (not the -1 other lanes see).
// The filename is built in the (global) slice — gpu_fopen reads its path via flat
// loads, so a private/stack buffer would marshal garbage. Inserted at kernel ENTRY.
extern "C" __device__ __noinline__ __attribute__((used))
void pw_open(void* slice) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
    if (pos != 0) return;
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* b = (char*)slice;
    int i = 16;                                        // name buffer at slice+16
    i = put_str(b, i, "wave_"); i = put_uint(b, i, wid); i = put_str(b, i, ".txt");
    b[i] = '\0';
    long long h = gpu_fopen(b + 16, "w");              // this wave's file
    *(volatile long long*)(b + 0) = h;                 // stash handle for pw_flush
    *(volatile int*)(b + 8)  = 0;                      // reset hits
    *(volatile int*)(b + 12) = 0;                      // reset lanes
}

// PER-WAVE FLUSH: read this wave's accumulated slice, format a human-readable line,
// and stream it to the trace file via gpu_fwrite (the fopen/fwrite hostcall path).
// Inserted at kernel EXIT, so each wave emits exactly one grouped line; the mailbox
// ticket lock serializes waves, so lines don't interleave. wid is the wave's logical
// flattened id, computed from the forwarded implicit ABI args (blockIdx/blockDim/
// threadIdx). Non-leaf (calls gpu_fwrite, has a local buffer) — inserted-call ABI.
extern "C" __device__ __noinline__ __attribute__((used))
void pw_flush(void* slice) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
    if (pos != 0) return;                             // one lane per wave writes the line
    char* b = (char*)slice;
    long long h  = *(volatile long long*)(b + 0);     // this wave's file handle (from pw_open)
    int hits     = *(volatile int*)(b + 8);
    int lanes    = *(volatile int*)(b + 12);          // read stats BEFORE reformatting
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    // Format into the (global) slice text area; gpu_fwrite reads its data via flat
    // loads, so it must be global, not a private/stack buffer.
    int i = 16;
    i = put_str(b, i, "wave ");    i = put_uint(b, i, wid);
    i = put_str(b, i, ": hits=");  i = put_uint(b, i, (unsigned)hits);
    i = put_str(b, i, " lanes=");  i = put_uint(b, i, (unsigned)lanes);
    b[i++] = '\n';
    gpu_fwrite(h, b + 16, i - 16);                    // -> this wave's own file
}

// ---- COMPOSABLE model: per-wave variable HOLDS a call's return value ----------
// These demonstrate the general "per-wave variable holds the return value, later
// passed into fwrite as its first parameter" idea WITHOUT a hand-written wrapper doing
// the store: dyninst inserts `hv = pw_openfile()` (the call SITE captures the ABI
// return into the per-wave var) and later `pw_writeln(hv.value())` (the var's held
// value passed as an argument). The wrappers just follow the ABI return contract.

// Returns a fixed sentinel in the ABI return registers (uniform) — isolates call-site
// return capture from any file/string dependency, for validating the mechanism.
extern "C" __device__ __noinline__ __attribute__((used))
int64_t pw_magic() {
    return 0xABCDE;
}

// Returns a file handle in the ABI return registers (gpu_fopen broadcasts it uniform).
extern "C" __device__ __noinline__ __attribute__((used))
int64_t pw_openfile() {
    return gpu_fopen("captured.txt", "w");
}

// Writes one line to the file identified by `handle` (supplied from a per-wave var's
// value()). Shows the held handle used as fwrite's first parameter.
extern "C" __device__ __noinline__ __attribute__((used))
void pw_writeln(int64_t handle) {
    gpu_fwrite_256(handle, "hello from a wave\n", 18);   // literal (device rodata): by-value
}

// ---- BB COUNTER (Dyninst-managed per-wave variable, slice passed by the call ABI) ----
// The per-wave slice base is delivered at LAUNCH time via the kernarg PerWaveBuf (a
// BPatch_perWaveVar): the mutator passes pw.address() (this wave's slice) as bb_inc's
// first argument, so Dyninst OWNS the variable and the ABI call hands its base pointer
// to the helper. One elected lane => per-wave BLOCK EXECUTIONS (no atomics). This is the
// sole per-wave model now (the old g_pw_base host-variable-define path was retired — a
// load-time symbol cannot carry a launch-time buffer address). Slice layout (STRIDE
// 4096): u32 counts[] at offset 0 (so a launcher slice[k] dump shows them); a text
// scratch area at BBI_TEXT_OFF for the flush, kept clear of counts[].
// (No fixed text offset: bb_flush_pw derives a COMPACT layout from nbb, so the per-wave
//  slice sizes to the actual block count instead of a fixed 4096 — see the arena.)

extern "C" __device__ __noinline__ __attribute__((used))
void bb_inc(void* base, int bbid) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
    if (pos != 0) return;                              // one lane/wave
    ((volatile unsigned*)base)[(unsigned)bbid] += 1u;  // counts[bbid] at slice+0
}

// EXIT flush for the bb_inc counter: read counts[] from THIS wave's slice base, write
// bbcount_<wid>.txt via the fopen/fwrite hostcalls. Slice-arg counterpart of the retired
// global-base bb_flush — the mutator passes the SAME pw.address() it passed to bb_inc.
extern "C" __device__ __noinline__ __attribute__((used))
void bb_flush_pw(void* base, int nbb) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
    if (pos != 0) return;
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* b = (char*)base;
    volatile unsigned* counts = (volatile unsigned*)b; // counts[0..nbb) at slice+0 (matches bb_inc)
    // COMPACT layout: counts end (8-aligned), then a 64B filename, then the report text.
    // The mutator sizes the per-wave arena to exactly this (nbb-derived), so no fixed slot.
    unsigned C = (((unsigned)(nbb > 0 ? nbb : 0) * 4u) + 7u) & ~7u;
    char* nm = b + C;                                  // filename [C, C+64)
    char* t  = b + C + 64u;                            // report text (flat-loadable global)
    int fi = 0;
    fi = put_str(nm, fi, "bbcount_"); fi = put_uint(nm, fi, wid); fi = put_str(nm, fi, ".txt"); nm[fi] = '\0';
    long long h = gpu_fopen(nm, "w");
    int i = 0;                                         // one line per block: "bb <k>: <count>\n"
    for (int k = 0; k < nbb; k++) {
        i = put_str(t, i, "bb ");  i = put_uint(t, i, (unsigned)k);
        i = put_str(t, i, ": ");   i = put_uint(t, i, counts[k]);
        t[i++] = '\n';
    }
    gpu_fwrite(h, t, i);                               // -> bbcount_<wid>.txt
}

// ---- REAL-FWRITE eval: pass-by-address vs by-value egress ----------------------
// Both take THIS wave's slice (a PerWaveBuf pointer; host-readable when the launcher's
// per-wave buffer is fine-grained/managed), open real_/bv_<wid>.txt, format an nbytes
// record into the slice, and stream it out. real_write uses gpu_real_fwrite (host reads
// the record by address — no 512B cap, one atomic host fwrite); bv_write uses gpu_fwrite
// (by-value copy into the ring — capped at 512B). Used to compare correctness + cost.
static __device__ int pwr_fill(char* rec, unsigned wid, int nbytes) {
    int n = 0; n = put_str(rec, n, "wave "); n = put_uint(rec, n, wid); n = put_str(rec, n, ": ");
    for (; n < nbytes - 1; n++) rec[n] = (char)('A' + (n % 26));
    rec[n++] = '\n';
    return n;
}
extern "C" __device__ __noinline__ __attribute__((used))
void real_write(void* slice, int nbytes) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    if (__builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo) != 0) return;   // one lane/wave
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* b = (char*)slice;
    int i = 0; i = put_str(b, i, "real_"); i = put_uint(b, i, wid); i = put_str(b, i, ".txt"); b[i] = '\0';
    long long h = gpu_fopen(b, "w");
    char* rec = b + 128;
    gpu_fwrite(h, rec, pwr_fill(rec, wid, nbytes));        // host reads `rec` by address (default)
}
extern "C" __device__ __noinline__ __attribute__((used))
void bv_write(void* slice, int nbytes) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    if (__builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo) != 0) return;   // one lane/wave
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* b = (char*)slice;
    int i = 0; i = put_str(b, i, "bv_"); i = put_uint(b, i, wid); i = put_str(b, i, ".txt"); b[i] = '\0';
    long long h = gpu_fopen(b, "w");
    char* rec = b + 128;
    gpu_fwrite_256(h, rec, pwr_fill(rec, wid, nbytes));    // by-value copy into the ring (≤512)
}
