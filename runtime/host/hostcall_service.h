// hostcall_service.h — the CPU-side ring service loop, shared by both host front-ends
// (the LD_PRELOAD product `preload.cpp` and the standalone diagnostic harness
// `launcher.cpp`). Both drive the SAME mailbox ABI, so the loop lives here once.
#pragma once
#include "../hostcalls.h"
#include <atomic>

// Service the GPU->CPU hostcall ring until *run becomes false. Scans all slots and
// services request-ready ones OUT OF ORDER (the property that avoids the FIFO
// ticket-lock deadlock at high wave counts). Files are keyed by handle so each wave can
// open its own file (gpu_fopen); a `primary` (first opened) backs the legacy WRITE_ID
// path. Files stay open until teardown. Prints a one-line summary on exit, prefixed by
// `tag` (e.g. "preload" / "host"). Honors HOSTCALL_VERBOSE for per-fopen logging.
void hostcall_service_loop(HostcallQueue* q, std::atomic<bool>& run, const char* tag);
