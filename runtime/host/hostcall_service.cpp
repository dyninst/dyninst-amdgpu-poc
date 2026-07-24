// hostcall_service.cpp — shared CPU-side ring service loop. See hostcall_service.h.
#include "hostcall_service.h"
#include <map>
#include <string>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

static bool verbose() { static bool v = getenv("HOSTCALL_VERBOSE") != nullptr; return v; }

// Service one request-ready slot in place (reads the request fields, writes the result).
static void service_slot(HostcallSlot* s, std::map<std::string, FILE*>& files,
                         FILE*& primary, long& opens, long& writes, long& closes) {
    // DIAGNOSTIC (HOSTCALL_NOOP): respond so the GPU's hc_call_and_wait completes, but do
    // NOT read the (possibly managed) record buffer or do host I/O. Isolates whether the
    // CPU touching the managed buffer mid-kernel is what faults the GPU (XNACK-off migration).
    static bool noop = getenv("HOSTCALL_NOOP") != nullptr;
    if (noop) {
        switch (s->opcode) {
        case HC_OP_FOPEN: s->handle = 1; s->retval = 0; break;          // dummy non-null handle
        case HC_OP_FWRITE: case HC_OP_FWRITE_PTR: case HC_OP_FREAD: s->retval = s->size; break;
        default: s->retval = 0; break;
        }
        return;                                                          // never touch bufptr/data/files
    }
    switch (s->opcode) {
    case HC_OP_FOPEN: {
        std::string name(s->path);
        FILE*& f = files[name];                          // open each distinct name once
        if (!f) {
            f = fopen(s->path, s->mode);
            if (!primary) primary = f;
            if (verbose()) fprintf(stderr, "[hostcall] fopen('%s','%s') -> %p\n",
                                   s->path, s->mode, (void*)f);
        }
        opens++;
        s->handle = (int64_t)(uintptr_t)f;               // return the handle to the device
        s->retval = f ? 0 : -1;
        break; }
    case HC_OP_FWRITE: {
        FILE* f = (FILE*)(uintptr_t)s->handle;            // per-wave file (from gpu_fopen)
        int n = f ? (int)fwrite(s->data, 1, s->size, f) : -1;
        if (f) fflush(f);
        writes++;
        s->retval = n;
        break; }
    case HC_OP_FWRITE_PTR: {                              // fwrite straight from the device VA
        FILE* f = (FILE*)(uintptr_t)s->handle;
        const void* p = (const void*)(uintptr_t)s->bufptr; // host-readable (managed/fine-grained)
        int n = (f && p) ? (int)fwrite(p, 1, s->size, f) : -1;
        if (f) fflush(f);
        writes++;
        s->retval = n;
        break; }
    case HC_OP_FREAD: {
        FILE* f = (FILE*)(uintptr_t)s->handle;
        s->retval = f ? (int)fread(s->data, 1, s->size, f) : -1;
        break; }
    case HC_OP_FCLOSE: {
        FILE* f = (FILE*)(uintptr_t)s->handle;
        if (f) fflush(f);                                 // defer real close to teardown
        else if (primary) fflush(primary);
        closes++;
        s->retval = 0;
        break; }
    case HC_OP_WRITE_ID:                                  // legacy per-site scalar id -> primary
        if (primary) { fprintf(primary, "[gpu] site %d\n", s->arg); fflush(primary); }
        writes++;
        s->retval = 0;
        break;
    default:
        s->retval = -1;
        break;
    }
}

void hostcall_service_loop(HostcallQueue* q, std::atomic<bool>& run, const char* tag) {
    std::map<std::string, FILE*> files;
    FILE* primary = nullptr;
    long opens = 0, writes = 0, closes = 0;
    while (run.load(std::memory_order_acquire)) {
        bool any = false;
        for (uint32_t i = 0; i < HC_NSLOTS; i++) {
            HostcallSlot* s = &q->slots[i];
            if (__atomic_load_n(&s->status, __ATOMIC_ACQUIRE) != 1) continue;
            any = true;
            service_slot(s, files, primary, opens, writes, closes);
            __atomic_thread_fence(__ATOMIC_RELEASE);
            __atomic_store_n(&s->status, 2, __ATOMIC_RELEASE);   // done -> release the wave
        }
        if (!any) std::this_thread::yield();
    }
    for (auto& kv : files) if (kv.second) fclose(kv.second);      // close every opened file
    fprintf(stderr, "[%s] serviced %ld fopen / %ld fwrite / %ld fclose (%zu distinct files)\n",
            tag, opens, writes, closes, files.size());
}
