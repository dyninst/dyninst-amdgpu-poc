#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
// preload.cpp — LD_PRELOAD front-end for the host hostcall runtime.
//
// Injects into a REAL HIP application and hooks the HSA loader path HIP uses internally.
// On ROCm 7.0.2 HIP loads each device code object as:
//     hsa_code_object_reader_create_from_memory(data, size)   <- the co bytes
//     hsa_executable_load_agent_code_object(exe, agent, reader)
//     hsa_executable_freeze(exe)
// We interpose those (plus __hipRegisterFatBinary and hipLaunchKernel) to substitute a
// pre-instrumented co, link the hostcall lib, service GPU->CPU hostcalls, and provision the
// per-wave buffer. The actual logic lives in hostcall_hooks.h (shared with the LD_AUDIT
// front-end audit.cpp); this file only supplies the LD_PRELOAD specifics:
//   * real_sym() -> dlsym(RTLD_NEXT): the real function is the NEXT one after our shadow.
//   * the exported interposer symbols, which the loader picks over the real ones because
//     this .so is preloaded; each just forwards to the matching hook body.
//
// Config (env): HOSTCALL_ORIG_CO / HOSTCALL_INST_CO / HOSTCALL_BUNDLE (or HOSTCALL_MANIFEST),
// HOSTCALL_LIB, HOSTCALL_VERBOSE. See experiments/*.sh.  (The per-wave STRIDE, the
// explicit-arg count, and the COV5 hidden-arg offsets are all read from the co itself —
// self-describing, no env: __dyninst_pw_stride symbol + the metadata .note.)

#include <dlfcn.h>
#define HLOG_PREFIX "[preload] "
#define HLOG_TAG    "preload"
#include "hostcall_hooks.h"

// LD_PRELOAD: the real function is the next definition in the search order after ours.
extern "C" void* real_sym(const char* name) { return dlsym(RTLD_NEXT, name); }

// Exported interposers: the dynamic linker binds callers to THESE (this .so is preloaded),
// and each forwards to the shared hook body. Signatures match the real functions exactly.
extern "C" void** __hipRegisterFatBinary(void* data) { return hook_register_fatbin(data); }

extern "C" hsa_status_t hsa_code_object_reader_create_from_memory(
        const void* data, size_t size, hsa_code_object_reader_t* reader) {
    return hook_co_reader(data, size, reader);
}
extern "C" hsa_status_t hsa_executable_load_agent_code_object(
        hsa_executable_t exe, hsa_agent_t agent, hsa_code_object_reader_t reader,
        const char* options, hsa_loaded_code_object_t* lco) {
    return hook_load(exe, agent, reader, options, lco);
}
extern "C" hsa_status_t hsa_executable_freeze(hsa_executable_t exe, const char* options) {
    return hook_freeze(exe, options);
}
extern "C" int hipLaunchKernel(const void* func, pw_dim3 numBlocks, pw_dim3 dimBlocks,
                               void** args, size_t sharedMemBytes, void* stream) {
    return hook_launch(func, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

__attribute__((destructor)) static void fini() { hostcall_teardown(); }
