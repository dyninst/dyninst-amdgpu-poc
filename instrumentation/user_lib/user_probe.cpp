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
    // Raw-slice style (the low-level escape hatch): this standalone marker keeps two
    // counters at slice+8/+12. The wired-up pw_open/pw_flush pair instead uses NAMED
    // per-wave variables the mutator declares — the preferred model.
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
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

// PER-WAVE OPEN / FLUSH: the mutator declares NAMED per-wave variables and hands each
// probe exactly the ones it needs — no ad-hoc byte layout the probe has to sub-divide.
// Three of them are genuine PERSISTENT per-wave state carried from the ENTRY probe to the
// EXIT probe: `handle` (this wave's open file), `hits`, and `lanes`. Two are transient
// staging: `name` (the filename pw_open builds for fopen) and `line` (the text pw_flush
// streams out). All live in the host-readable per-wave arena, so gpu_fopen/gpu_fwrite —
// which marshal their string/record args by flat loads from that VA — read them correctly.

// PER-WAVE OPEN: each wave opens its OWN file "wave_<wid>.txt" and stashes the returned
// handle in its `handle` variable, so the exit probe writes to that file. gpu_fopen returns
// the handle only on its elected lane; we do the format+open+store under one elected lane so
// the stored handle is the real one (not the -1 other lanes see). Inserted at kernel ENTRY.
extern "C" __device__ __noinline__ __attribute__((used))
void pw_open(void* handle_, void* hits_, void* lanes_, void* name_) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
    if (pos != 0) return;
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* nm = (char*)name_;                           // filename staging buffer
    int i = 0;
    i = put_str(nm, i, "wave_"); i = put_uint(nm, i, wid); i = put_str(nm, i, ".txt");
    nm[i] = '\0';
    long long h = gpu_fopen(nm, "w");                  // this wave's file
    *(volatile long long*)handle_ = h;                 // stash handle for pw_flush
    *(volatile int*)hits_  = 0;                         // reset per-wave accumulators
    *(volatile int*)lanes_ = 0;
}

// PER-WAVE FLUSH: read this wave's persistent state (handle/hits/lanes stashed by pw_open),
// format a human-readable line into the `line` buffer, and stream it to the trace file via
// gpu_fwrite. Inserted at kernel EXIT, so each wave emits exactly one grouped line; the
// mailbox ticket lock serializes waves, so lines don't interleave. Non-leaf (calls
// gpu_fwrite) — inserted-call ABI.
extern "C" __device__ __noinline__ __attribute__((used))
void pw_flush(void* handle_, void* hits_, void* lanes_, void* line_) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    unsigned pos = __builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo);
    if (pos != 0) return;                             // one lane per wave writes the line
    long long h  = *(volatile long long*)handle_;     // this wave's file handle (from pw_open)
    int hits     = *(volatile int*)hits_;
    int lanes    = *(volatile int*)lanes_;            // read stats BEFORE reformatting
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* t = (char*)line_;                           // output-line staging buffer
    int i = 0;
    i = put_str(t, i, "wave ");    i = put_uint(t, i, wid);
    i = put_str(t, i, ": hits=");  i = put_uint(t, i, (unsigned)hits);
    i = put_str(t, i, " lanes=");  i = put_uint(t, i, (unsigned)lanes);
    t[i++] = '\n';
    gpu_fwrite(h, t, i);                              // -> this wave's own file
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
// LAYOUT-AGNOSTIC: the mutator hands three DISTINCT per-wave variables — the counts array
// bb_inc wrote, a filename staging buffer, and a report-text staging buffer — so the probe
// does NO offset arithmetic and makes no assumptions about how they're packed. (The
// filename/report buffers must be in the host-readable per-wave arena: gpu_fwrite is
// pass-by-address, so the host reads `report` straight from that device VA.)
void bb_flush_pw(void* counts_, void* fname_, void* report_, int nbb) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo  = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    if (__builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo) != 0) return;   // one lane/wave
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    volatile unsigned* counts = (volatile unsigned*)counts_;   // written by bb_inc / bb_inc_one
    char* nm = (char*)fname_;                                  // filename staging buffer
    char* t  = (char*)report_;                                 // report-text staging buffer
    int fi = 0;
    fi = put_str(nm, fi, "bbcount_"); fi = put_uint(nm, fi, wid); fi = put_str(nm, fi, ".txt"); nm[fi] = '\0';
    long long h = gpu_fopen(nm, "w");
    int i = 0;                                                 // one line per block: "bb <k>: <count>\n"
    for (int k = 0; k < nbb; k++) {
        i = put_str(t, i, "bb ");  i = put_uint(t, i, (unsigned)k);
        i = put_str(t, i, ": ");   i = put_uint(t, i, counts[k]);
        t[i++] = '\n';
    }
    gpu_fwrite(h, t, i);                                       // -> bbcount_<wid>.txt
}

// ---- REAL-FWRITE eval: pass-by-address vs by-value egress ----------------------
// Both take TWO distinct per-wave variables the mutator declares — a filename staging
// buffer and the nbytes record buffer (both host-readable when the per-wave buffer is
// fine-grained/managed) — so the probe does no offset math. They open real_/bv_<wid>.txt,
// format the record, and stream it out. real_write uses gpu_fwrite pass-by-address (host
// reads the record by address — no 512B cap, one atomic host fwrite); bv_write uses
// gpu_fwrite_256 (by-value copy into the ring — capped at 512B). Compare correctness + cost.
static __device__ int pwr_fill(char* rec, unsigned wid, int nbytes) {
    int n = 0; n = put_str(rec, n, "wave "); n = put_uint(rec, n, wid); n = put_str(rec, n, ": ");
    for (; n < nbytes - 1; n++) rec[n] = (char)('A' + (n % 26));
    rec[n++] = '\n';
    return n;
}
extern "C" __device__ __noinline__ __attribute__((used))
void real_write(void* fname_, void* record_, int nbytes) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    if (__builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo) != 0) return;   // one lane/wave
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* nm = (char*)fname_;                               // filename staging buffer
    char* rec = (char*)record_;                             // record staging buffer
    int i = 0; i = put_str(nm, i, "real_"); i = put_uint(nm, i, wid); i = put_str(nm, i, ".txt"); nm[i] = '\0';
    long long h = gpu_fopen(nm, "w");
    gpu_fwrite(h, rec, pwr_fill(rec, wid, nbytes));        // host reads `rec` by address (default)
}
extern "C" __device__ __noinline__ __attribute__((used))
void bv_write(void* fname_, void* record_, int nbytes) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    if (__builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo) != 0) return;   // one lane/wave
    unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / 64u;
    char* nm = (char*)fname_;                               // filename staging buffer
    char* rec = (char*)record_;                             // record staging buffer
    int i = 0; i = put_str(nm, i, "bv_"); i = put_uint(nm, i, wid); i = put_str(nm, i, ".txt"); nm[i] = '\0';
    long long h = gpu_fopen(nm, "w");
    gpu_fwrite_256(h, rec, pwr_fill(rec, wid, nbytes));    // by-value copy into the ring (≤512)
}

// ---- MULTI-VAR test: two independent per-wave variables ------------------------
// Dyninst passes THIS wave's slice for EACH of two per-wave variables at distinct arena
// offsets — a = base + wid*STRIDE + off_a, b = base + wid*STRIDE + off_b. We write a
// distinct marker into each; the host then confirms they landed at different addresses,
// proving emitCall's per-arg offset add (the multi-var lowering). One elected lane/wave.
// ---- BB COUNTER, one-var-per-block style ---------------------------------------
// The multi-var alternative to bb_inc(base, bbid): the mutator declares one per-wave
// variable PER basic block and passes THIS block's counter pointer directly, so the
// probe needs no bbid and no indexing — it just bumps the cell it was handed. One
// elected lane/wave => per-wave block-execution counts. The flush is unchanged
// (bb_flush_pw walks the contiguous counters from the slice base).
extern "C" __device__ __noinline__ __attribute__((used))
void bb_inc_one(void* counter) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    if (__builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo) != 0) return;   // one lane/wave
    *(volatile unsigned*)counter += 1u;
}

extern "C" __device__ __noinline__ __attribute__((used))
void pw_mark2(void* a, void* b) {
    unsigned long long ex = __builtin_amdgcn_read_exec();
    unsigned lo = __builtin_amdgcn_mbcnt_lo((unsigned)ex, 0u);
    if (__builtin_amdgcn_mbcnt_hi((unsigned)(ex >> 32), lo) != 0) return;   // one lane/wave
    *(volatile unsigned*)a = 0xAAAAu;
    *(volatile unsigned*)b = 0xBBBBu;
}
