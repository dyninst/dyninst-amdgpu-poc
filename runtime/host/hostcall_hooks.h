#pragma once
// hostcall_hooks.h — the loader-interposition logic shared by the two host injectors:
//   preload.cpp  (LD_PRELOAD: symbols shadow the real ones; real = dlsym(RTLD_NEXT))
//   audit.cpp    (LD_AUDIT:  la_symbind redirects to these bodies; real = dlsym(link_map))
//
// Both drive the SAME flow into a real HIP application:
//   1. __hipRegisterFatBinary : swap the app's fatbin for the pre-instrumented bundle.
//   2. code_object_reader_create_from_memory : detect the instrumented co being loaded.
//   3. executable_load_agent_code_object : augment that exe once — define `mailbox`,
//      load the hostcall lib — then start the CPU service thread.
//   4. executable_freeze : (log) cross-object relocs resolve here.
//   5. hipLaunchKernel : allocate the managed per-wave buffer + append it as a kernarg.
//
// The ONLY thing that differs between the two injectors is how a REAL function address is
// obtained. That is abstracted behind `real_sym(name)`, which each .cpp defines; every HSA
// and HIP call in here is routed through it via BIND(). This is mandatory for LD_AUDIT (a
// direct call would bind in the audit namespace and pull in a second libhsa) and is exactly
// equivalent to the old direct calls under LD_PRELOAD (RTLD_NEXT resolves the real lib).
//
// NOTE: include this from EXACTLY ONE translation unit per shared object (it defines
// globals, not just declarations). Define HLOG_PREFIX before including to tag the logs.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <dlfcn.h>
#include <atomic>
#include <set>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "../hostcalls.h"      // shared mailbox ABI
#include "hostcall_service.h"  // shared CPU ring service loop
#include "co_stride.h"         // read __dyninst_pw_stride from the instrumented co

#ifndef HLOG_PREFIX
#define HLOG_PREFIX "[hostcall] "   // fprintf line prefix (already bracketed)
#endif
#ifndef HLOG_TAG
#define HLOG_TAG "hostcall"         // bare tag for the service loop (it wraps as "[%s]")
#endif

// Each injector defines this: the address of the REAL (app-namespace) function `name`.
//   preload: dlsym(RTLD_NEXT, name)          audit: dlsym((void*)libmap, name)
extern "C" void* real_sym(const char* name);

// Resolve the real once, cache it. Uses real_sym so both injectors share every call site.
#define BIND(name) \
    static decltype(&name) real_##name = (decltype(&name))real_sym(#name)

// HOSTCALL_VERBOSE gates the chatty logs. Lazy (a file-scope dynamic initializer is not
// guaranteed to run inside an LD_AUDIT module — its ELF init/array is not invoked).
static bool hc_verbose() { static bool v = getenv("HOSTCALL_VERBOSE") != nullptr; return v; }
#define LOG(...) do { if (hc_verbose()) fprintf(stderr, __VA_ARGS__); } while (0)

// ------------------------------------------------------------------ globals
// Container globals are function-local statics (constructed on first use via a guard),
// NOT file-scope objects: an LD_AUDIT module's static constructors may never run, which
// would leave a file-scope std::set/std::vector zero- but not constructed. (This also
// dodges the static-init-order fiasco with __hipRegisterFatBinary firing during an early
// DSO init.) Trivially/constexpr-constructible globals below are safe zero-initialized.
static std::mutex        g_mtx;
static HostcallMailbox*  g_mbox = nullptr;   // one shared mailbox (host-coherent)
static std::thread       g_svc;
static std::atomic<bool> g_run{true};
static std::atomic<bool> g_svc_started{false};

static std::set<uint64_t>& g_inst_readers()   { static std::set<uint64_t> s; return s; }  // readers we substituted
static std::set<uint64_t>& g_augmented_exes() { static std::set<uint64_t> s; return s; }  // exes given lib+mailbox

// One code-object substitution: match the app's original co, swap in the instrumented bundle.
struct Sub { std::vector<uint8_t> orig, inst, bundle; std::string tag; };
static std::vector<Sub>&     g_subs()   { static std::vector<Sub> v; return v; }
static std::vector<uint8_t>& g_lib_co() { static std::vector<uint8_t> v; return v; }

// HIP's fat-binary wrapper: binary points at a __CLANG_OFFLOAD_BUNDLE__ HIP parses AND loads.
struct FatBinaryWrapper { unsigned magic; unsigned version; void* binary; void* dummy; };
extern "C" void** __hipRegisterFatBinary(void* data);   // HIP C ABI (for BIND's decltype)

// ------------------------------------------------------------------ file slurp
static bool slurp(const char* path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, HLOG_PREFIX "cannot open %s: %s\n", path, strerror(errno)); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(n);
    size_t got = fread(out.data(), 1, n, f);
    fclose(f);
    if ((long)got != n) { fprintf(stderr, HLOG_PREFIX "short read %s\n", path); return false; }
    return true;
}

static bool add_sub(const char* orig, const char* inst, const char* bundle) {
    Sub s; s.tag = orig;
    if (!slurp(orig, s.orig) || !slurp(inst, s.inst) || !slurp(bundle, s.bundle)) return false;
    fprintf(stderr, HLOG_PREFIX "sub[%zu]: %s (%zu B) -> inst %zu B / bundle %zu B\n",
            g_subs().size(), orig, s.orig.size(), s.inst.size(), s.bundle.size());
    g_subs().push_back(std::move(s));
    return true;
}
static bool load_config() {
    const char* lib = getenv("HOSTCALL_LIB");
    if (!lib || !slurp(lib, g_lib_co())) {
        fprintf(stderr, HLOG_PREFIX "set HOSTCALL_LIB\n"); return false;
    }
    if (const char* mf = getenv("HOSTCALL_MANIFEST")) {           // "orig inst bundle" lines
        FILE* f = fopen(mf, "r"); if (!f) { perror(mf); return false; }
        char a[1024], b[1024], c[1024];
        while (fscanf(f, "%1023s %1023s %1023s", a, b, c) == 3) add_sub(a, b, c);
        fclose(f);
    } else {
        const char* orig = getenv("HOSTCALL_ORIG_CO");
        const char* inst = getenv("HOSTCALL_INST_CO");
        const char* bundle = getenv("HOSTCALL_BUNDLE");
        if (!orig || !inst || !bundle) {
            fprintf(stderr, HLOG_PREFIX "set HOSTCALL_ORIG_CO/INST_CO/BUNDLE (or HOSTCALL_MANIFEST)\n");
            return false;
        }
        add_sub(orig, inst, bundle);
    }
    fprintf(stderr, HLOG_PREFIX "%zu substitution(s); lib=%s (%zu B)\n", g_subs().size(), lib, g_lib_co().size());
    return !g_subs().empty();
}
static std::once_flag g_cfg_once;
static bool           g_cfg_ok_v = false;
static bool cfg() { std::call_once(g_cfg_once, []{ g_cfg_ok_v = load_config(); }); return g_cfg_ok_v; }

// ------------------------------------------------------------------ bundle parse
static bool bundle_device_co(const void* b, const uint8_t*& co, uint64_t& co_sz, uint64_t& total) {
    const uint8_t* p = (const uint8_t*)b;
    if (memcmp(p, "__CLANG_OFFLOAD_BUNDLE__", 24) != 0) return false;
    uint64_t n; memcpy(&n, p + 24, 8);
    const uint8_t* e = p + 32;
    co = nullptr; total = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t off, sz, tl; memcpy(&off,e,8); memcpy(&sz,e+8,8); memcpy(&tl,e+16,8);
        const char* triple = (const char*)(e + 24);
        if (off + sz > total) total = off + sz;
        if (tl >= 6 && memmem(triple, tl, "amdgcn", 6)) { co = p + off; co_sz = sz; }
        e += 24 + tl;
    }
    return co != nullptr;
}

// ------------------------------------------------------------------ HSA discovery
struct AgentSearch { hsa_agent_t agent; bool found; };
static hsa_status_t find_cpu(hsa_agent_t a, void* d) {
    BIND(hsa_agent_get_info);
    hsa_device_type_t t; real_hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_CPU) { auto* s=(AgentSearch*)d; s->agent=a; s->found=true; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t find_gpu(hsa_agent_t a, void* d) {
    BIND(hsa_agent_get_info);
    hsa_device_type_t t; real_hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU) { auto* s=(AgentSearch*)d; s->agent=a; s->found=true; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}
struct PoolSearch { hsa_amd_memory_pool_t pool; bool found; };
static hsa_status_t find_fine_grained(hsa_amd_memory_pool_t pool, void* d) {
    BIND(hsa_amd_memory_pool_get_info);
    hsa_amd_segment_t seg;
    real_hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags;
    real_hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
        auto* s=(PoolSearch*)d; s->pool=pool; s->found=true; return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

// ------------------------------------------------------------------ lazy runtime init
// Allocate the shared mailbox in host-coherent fine-grained memory, grant the GPU agent
// access, and start the service thread. Once, on first instrumented load.
static bool ensure_runtime(hsa_agent_t gpu) {
    if (g_svc_started.load()) return g_mbox != nullptr;
    BIND(hsa_iterate_agents);
    BIND(hsa_amd_agent_iterate_memory_pools);
    BIND(hsa_amd_memory_pool_allocate);
    BIND(hsa_amd_agents_allow_access);

    AgentSearch cpu{}; real_hsa_iterate_agents(find_cpu, &cpu);
    if (!cpu.found) { fprintf(stderr, HLOG_PREFIX "no CPU agent\n"); return false; }
    PoolSearch fg{}; real_hsa_amd_agent_iterate_memory_pools(cpu.agent, find_fine_grained, &fg);
    if (!fg.found) { fprintf(stderr, HLOG_PREFIX "no fine-grained pool\n"); return false; }

    if (real_hsa_amd_memory_pool_allocate(fg.pool, sizeof(HostcallMailbox), 0, (void**)&g_mbox)
            != HSA_STATUS_SUCCESS) { fprintf(stderr, HLOG_PREFIX "mailbox alloc failed\n"); return false; }
    real_hsa_amd_agents_allow_access(1, &gpu, nullptr, g_mbox);
    memset(g_mbox, 0, sizeof(*g_mbox));
    for (uint32_t i = 0; i < HC_NSLOTS; i++) g_mbox->slots[i].turn = i;  // ring generation gates

    g_run.store(true);
    g_svc = std::thread(hostcall_service_loop, g_mbox, std::ref(g_run), HLOG_TAG);
    g_svc_started.store(true);
    fprintf(stderr, HLOG_PREFIX "mailbox @ %p; service thread started\n", (void*)g_mbox);
    return true;
}

// Define `mailbox` and load the hostcall library into `exe`, once per executable. Uses the
// REAL loader entry points (via real_sym) so we don't recurse into our own interposition.
static void augment_executable(hsa_executable_t exe, hsa_agent_t agent) {
    BIND(hsa_executable_agent_global_variable_define);
    BIND(hsa_code_object_reader_create_from_memory);
    BIND(hsa_executable_load_agent_code_object);

    hsa_status_t s = real_hsa_executable_agent_global_variable_define(exe, agent, "mailbox", g_mbox);
    if (s != HSA_STATUS_SUCCESS) fprintf(stderr, HLOG_PREFIX "define(mailbox) status=%d\n", s);

    hsa_code_object_reader_t r{};
    s = real_hsa_code_object_reader_create_from_memory(g_lib_co().data(), g_lib_co().size(), &r);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, HLOG_PREFIX "lib reader status=%d\n", s); return; }
    s = real_hsa_executable_load_agent_code_object(exe, agent, r, "", nullptr);
    if (s != HSA_STATUS_SUCCESS) fprintf(stderr, HLOG_PREFIX "load(lib) status=%d\n", s);
    else fprintf(stderr, HLOG_PREFIX "augmented exe=%lu: mailbox defined + hostcall lib loaded\n", exe.handle);
}

// ================================================================== HOOK BODIES
// Each has the EXACT ABI of the function it stands in for, so an injector can expose it as
// an interposer (LD_PRELOAD) or hand its address to la_symbind (LD_AUDIT) unchanged.

// __hipRegisterFatBinary: swap the app fatbin for our instrumented bundle so HIP caches the
// instrumented KD's metadata AND loads it. Match by embedded device co == original app co.
static void** hook_register_fatbin(void* data) {
    BIND(__hipRegisterFatBinary);
    if (cfg() && data) {
        auto* w = (FatBinaryWrapper*)data;
        const uint8_t* co; uint64_t co_sz, total;
        if (w->binary && bundle_device_co(w->binary, co, co_sz, total)) {
            for (auto& s : g_subs()) {
                if (co_sz == s.orig.size() && memcmp(co, s.orig.data(), co_sz) == 0) {
                    FatBinaryWrapper* nw = new FatBinaryWrapper(*w);
                    nw->binary = s.bundle.data();
                    fprintf(stderr, HLOG_PREFIX "__hipRegisterFatBinary: SUBSTITUTED %s "
                            "(device co %lu B -> instrumented bundle %zu B)\n",
                            s.tag.c_str(), co_sz, s.bundle.size());
                    return real___hipRegisterFatBinary(nw);
                }
            }
        }
    }
    return real___hipRegisterFatBinary(data);
}

// Detector: the substituted fatbin makes HIP hand us the instrumented co bytes here — mark
// that reader so the load hook augments its executable.
static hsa_status_t hook_co_reader(const void* data, size_t size, hsa_code_object_reader_t* reader) {
    BIND(hsa_code_object_reader_create_from_memory);
    hsa_status_t s = real_hsa_code_object_reader_create_from_memory(data, size, reader);
    bool matched = false;
    if (s == HSA_STATUS_SUCCESS && cfg()) {
        for (auto& sub : g_subs()) {
            if (size == sub.inst.size() && memcmp(data, sub.inst.data(), size) == 0) {
                std::lock_guard<std::mutex> lk(g_mtx);
                g_inst_readers().insert(reader->handle);
                fprintf(stderr, HLOG_PREFIX "detected instrumented co load %s (%zu B); reader=%lu\n",
                        sub.tag.c_str(), size, reader->handle);
                matched = true; break;
            }
        }
    }
    if (!matched) LOG(HLOG_PREFIX "passthrough co_reader size=%zu\n", size);
    return s;
}

static hsa_status_t hook_load(hsa_executable_t exe, hsa_agent_t agent,
                              hsa_code_object_reader_t reader, const char* options,
                              hsa_loaded_code_object_t* lco) {
    BIND(hsa_executable_load_agent_code_object);
    bool is_inst;
    { std::lock_guard<std::mutex> lk(g_mtx); is_inst = g_inst_readers().count(reader.handle) > 0; }
    if (is_inst) {                             // make the exe self-contained before loading it
        if (ensure_runtime(agent)) {
            std::lock_guard<std::mutex> lk(g_mtx);
            if (g_augmented_exes().insert(exe.handle).second)
                augment_executable(exe, agent);
        }
    }
    hsa_status_t s = real_hsa_executable_load_agent_code_object(exe, agent, reader, options, lco);
    if (is_inst) LOG(HLOG_PREFIX "loaded instrumented app co into exe=%lu status=%d\n", exe.handle, s);
    return s;
}

static hsa_status_t hook_freeze(hsa_executable_t exe, const char* options) {
    BIND(hsa_executable_freeze);
    hsa_status_t s = real_hsa_executable_freeze(exe, options);
    { std::lock_guard<std::mutex> lk(g_mtx);
      if (g_augmented_exes().count(exe.handle))
          fprintf(stderr, HLOG_PREFIX "froze augmented exe=%lu status=%d (cross-object relocs resolved)\n",
                  exe.handle, s); }
    return s;
}

// ------------------------------------------------------------------ launch hook
namespace { struct pw_dim3 { uint32_t x, y, z; }; }   // dim3 ABI: {uint32 x,y,z}
extern "C" int hipMallocManaged(void** ptr, size_t size, unsigned int flags);
extern "C" int hipMalloc(void** ptr, size_t size);
extern "C" int hipLaunchKernel(const void* func, pw_dim3 numBlocks, pw_dim3 dimBlocks,
                               void** args, size_t sharedMemBytes, void* stream);

// Per-wave STRIDE (bytes/wave). Self-describing: read from __dyninst_pw_stride in the
// instrumented co (HOSTCALL_INST_CO). PW_STRIDE env overrides; 4096 fallback.
static uint32_t pw_stride() {
    static uint32_t s = [] {
        if (const char* e = getenv("PW_STRIDE")) return (uint32_t)strtoul(e, nullptr, 0);
        uint32_t v = co_read_symbol_u32(getenv("HOSTCALL_INST_CO"), "__dyninst_pw_stride", 4096);
        fprintf(stderr, HLOG_PREFIX "per-wave STRIDE = %u B (from __dyninst_pw_stride)\n", v);
        return v;
    }();
    return s;
}
static void* g_arg_buf = nullptr;   // this launch's per-wave arg buffer (device)

// Append a per-wave buffer pointer as an extra EXPLICIT kernel argument. Sized HERE, at
// launch, from the actual wave count (nwaves * STRIDE). PW_NARGS = the kernel's ORIGINAL
// explicit-arg count (hipLaunchKernel's args[] has no length). PW_ALLOC picks the memory
// type (managed default). Reaches the kernarg only when the co carries the extra descriptor.
static int hook_launch(const void* func, pw_dim3 numBlocks, pw_dim3 dimBlocks,
                       void** args, size_t sharedMemBytes, void* stream) {
    BIND(hipLaunchKernel);
    const char* ne = getenv("PW_NARGS");
    if (ne && args) {
        int n = atoi(ne);
        uint64_t wpb    = ((uint64_t)dimBlocks.x * dimBlocks.y * dimBlocks.z + 63) / 64;  // waves/block
        uint64_t nwaves = (uint64_t)numBlocks.x * numBlocks.y * numBlocks.z * wpb;
        size_t   sz     = (size_t)nwaves * pw_stride();
        const char* alloc = getenv("PW_ALLOC"); if (!alloc) alloc = "managed";
        int rc = 0;
        if (!strcmp(alloc, "device")) {
            BIND(hipMalloc);
            rc = real_hipMalloc(&g_arg_buf, sz);                            // coarse-grained device
        } else if (!strcmp(alloc, "fg")) {                                 // HSA fine-grained
            BIND(hsa_iterate_agents); BIND(hsa_amd_agent_iterate_memory_pools);
            BIND(hsa_amd_memory_pool_allocate); BIND(hsa_amd_agents_allow_access);
            static hsa_amd_memory_pool_t s_fg{}; static hsa_agent_t s_gpu{}; static bool s_ok=false;
            if (!s_ok) { AgentSearch cpu{}, gpu{}; real_hsa_iterate_agents(find_cpu,&cpu); real_hsa_iterate_agents(find_gpu,&gpu);
                PoolSearch fg{}; if (cpu.found) real_hsa_amd_agent_iterate_memory_pools(cpu.agent, find_fine_grained, &fg);
                s_fg=fg.pool; s_gpu=gpu.agent; s_ok=cpu.found&&gpu.found&&fg.found; }
            rc = (s_ok && real_hsa_amd_memory_pool_allocate(s_fg, sz, 0, &g_arg_buf)==HSA_STATUS_SUCCESS) ? 0 : 1;
            if (!rc) real_hsa_amd_agents_allow_access(1, &s_gpu, nullptr, g_arg_buf);
        } else {
            BIND(hipMallocManaged);
            rc = real_hipMallocManaged(&g_arg_buf, sz, 1u);                 // managed (default)
        }
        fprintf(stderr, HLOG_PREFIX "PW_ALLOC=%s\n", alloc);
        if (rc != 0 || !g_arg_buf) {
            fprintf(stderr, HLOG_PREFIX "per-wave alloc(%s,%zu) failed; launching unmodified\n", alloc, sz);
            return real_hipLaunchKernel(func, numBlocks, dimBlocks, args, sharedMemBytes, stream);
        }
        std::vector<void*> na(args, args + n);
        na.push_back(&g_arg_buf);     // HIP copies *(&g_arg_buf) = the device ptr into the kernarg
        fprintf(stderr, HLOG_PREFIX "hipLaunchKernel: per-wave arg buf %p (%lu waves x %u = %zu B)"
                        " appended as explicit arg[%d]\n", g_arg_buf, (unsigned long)nwaves,
                        pw_stride(), sz, n);
        return real_hipLaunchKernel(func, numBlocks, dimBlocks, na.data(), sharedMemBytes, stream);
    }
    return real_hipLaunchKernel(func, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

// ------------------------------------------------------------------ teardown
static void hostcall_teardown() {
    if (g_svc_started.load()) {
        g_run.store(false, std::memory_order_release);
        if (g_svc.joinable()) g_svc.join();
    }
}
