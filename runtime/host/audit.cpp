#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
// audit.cpp — LD_AUDIT front-end for the host hostcall runtime (rtld-audit interface).
//
// Same job as preload.cpp, driven through the dynamic linker's AUDIT interface instead of
// symbol preloading. Set LD_AUDIT=audit.so and the linker calls our la_* callbacks; in
// la_symbind64 we return the address of a hook body (from hostcall_hooks.h) for each of the
// five entry points, which redirects the caller's PLT slot to us — the audit equivalent of
// an LD_PRELOAD shadow. The hook logic is identical; only the plumbing differs.
//
// Two consequences of the audit design shape this file (see hostcall_hooks.h):
//   1. SEPARATE NAMESPACE. The audit module is loaded in its own link-map namespace, so it
//      must NOT link libhsa/libamdhip (that would load a second copy). It calls the app's
//      real functions only through pointers resolved with dlsym() against the app's already-
//      loaded libhsa/libamdhip link_maps — captured here in la_objopen and used by real_sym.
//   2. NO MODULE CONSTRUCTORS. glibc does not run an audit module's ELF init/finalize, so
//      teardown is registered with atexit() (in la_version) rather than a destructor, and
//      the shared header keeps its nontrivial state in lazily-constructed function statics.
//
// Config env is identical to the LD_PRELOAD path (HOSTCALL_ORIG_CO/INST_CO/BUNDLE, ...).

#include <link.h>
#include <dlfcn.h>
#include <elf.h>
#include <cstring>

#define HLOG_PREFIX "[audit] "
#define HLOG_TAG    "audit"
#include "hostcall_hooks.h"

// The app-namespace objects that DEFINE the functions we call/redirect. Captured as
// link_maps in la_objopen and resolved out of the app's own copies (never the audit
// namespace, which must not hold a second HSA runtime).
static struct link_map* g_hsa_map = nullptr;   // libhsa-runtime64  (hsa_*)
static struct link_map* g_hip_map = nullptr;   // libamdhip64       (hip*, __hipRegister*)

// Resolve `name` in a loaded object by walking ITS OWN dynamic symbol table. We can NOT use
// dlsym((void*)link_map): passing a raw audit link_map to dlsym crashes in glibc's scope
// walk (dl-lookup do_lookup_x), because these maps aren't dlopen handles. Reading .dynsym
// directly sidesteps all of that. glibc relocates the DT_* address entries in l_ld in place
// (adds the load bias); we defensively add l_addr to anything still below it (and to each
// symbol's st_value, which is never pre-adjusted).
static void* elf_dynsym(struct link_map* m, const char* name) {
    if (!m || !m->l_ld) return nullptr;
    auto abs = [&](ElfW(Addr) p) -> ElfW(Addr) { return p < m->l_addr ? p + m->l_addr : p; };
    const char*       strtab = nullptr;
    const ElfW(Sym)*  symtab = nullptr;
    const uint32_t*   hash   = nullptr;   // DT_HASH (gives symbol count directly)
    const uint32_t*   gnu    = nullptr;   // DT_GNU_HASH (fallback)
    for (ElfW(Dyn)* d = m->l_ld; d->d_tag != DT_NULL; ++d) {
        switch (d->d_tag) {
            case DT_STRTAB:   strtab = (const char*)      abs(d->d_un.d_ptr); break;
            case DT_SYMTAB:   symtab = (const ElfW(Sym)*) abs(d->d_un.d_ptr); break;
            case DT_HASH:     hash   = (const uint32_t*)  abs(d->d_un.d_ptr); break;
            case DT_GNU_HASH: gnu    = (const uint32_t*)  abs(d->d_un.d_ptr); break;
        }
    }
    if (!strtab || !symtab) return nullptr;

    // Symbol count: DT_HASH.nchain is exact; else derive the max index from DT_GNU_HASH.
    uint32_t nsym = 0;
    if (hash) {
        nsym = hash[1];
    } else if (gnu) {
        uint32_t nbuckets = gnu[0], symbias = gnu[1], bloom_size = gnu[2];
        const ElfW(Addr)* bloom = (const ElfW(Addr)*)(gnu + 4);
        const uint32_t*   buckets = (const uint32_t*)(bloom + bloom_size);
        const uint32_t*   chain   = buckets + nbuckets;
        uint32_t last = 0;
        for (uint32_t i = 0; i < nbuckets; i++) if (buckets[i] > last) last = buckets[i];
        if (last < symbias) return nullptr;
        while (!(chain[last - symbias] & 1)) last++;   // walk to end of the chain
        nsym = last + 1;
    } else return nullptr;

    for (uint32_t i = 0; i < nsym; i++) {
        const ElfW(Sym)& s = symtab[i];
        if (s.st_name && s.st_value && !strcmp(strtab + s.st_name, name))
            return (void*)abs(s.st_value);
    }
    return nullptr;
}

// real_sym for LD_AUDIT: resolve in the app's libhsa first, then libamdhip.
extern "C" void* real_sym(const char* name) {
    void* p = elf_dynsym(g_hsa_map, name);
    return p ? p : elf_dynsym(g_hip_map, name);
}

// ------------------------------------------------------------------ rtld-audit callbacks
extern "C" unsigned int la_version(unsigned int version) {
    // Teardown must be arranged here: an audit module's fini is not called by the linker.
    atexit(hostcall_teardown);
    return version <= LAV_CURRENT ? version : LAV_CURRENT;
}

// Called once per loaded object. Capture the two libraries we resolve against, and request
// PLT auditing (BINDFROM on callers, BINDTO on definers) so la_symbind fires for every
// binding — cheap here since symbind only does a handful of string compares.
extern "C" unsigned int la_objopen(struct link_map* map, Lmid_t /*lmid*/, uintptr_t* /*cookie*/) {
    const char* n = map->l_name ? map->l_name : "";
    if (strstr(n, "libhsa-runtime64")) { g_hsa_map = map; LOG(HLOG_PREFIX "objopen libhsa: %s\n", n); }
    if (strstr(n, "libamdhip64"))      { g_hip_map = map; LOG(HLOG_PREFIX "objopen libhip: %s\n", n); }
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

// The interposition point: return the address of our hook body for a symbol we intercept,
// which redirects the caller's PLT slot to it; return the real address (sym->st_value) for
// everything else. Same five entry points as the LD_PRELOAD interposers.
extern "C" uintptr_t la_symbind64(Elf64_Sym* sym, unsigned int /*ndx*/,
                                  uintptr_t* /*refcook*/, uintptr_t* /*defcook*/,
                                  unsigned int* /*flags*/, const char* symname) {
    if (!strcmp(symname, "__hipRegisterFatBinary"))                    return (uintptr_t)&hook_register_fatbin;
    if (!strcmp(symname, "hsa_code_object_reader_create_from_memory")) return (uintptr_t)&hook_co_reader;
    if (!strcmp(symname, "hsa_executable_load_agent_code_object"))     return (uintptr_t)&hook_load;
    if (!strcmp(symname, "hsa_executable_freeze"))                     return (uintptr_t)&hook_freeze;
    if (!strcmp(symname, "hipLaunchKernel"))                           return (uintptr_t)&hook_launch;
    return sym->st_value;
}
