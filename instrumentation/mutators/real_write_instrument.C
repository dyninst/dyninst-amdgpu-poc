/*
 *  real_write_instrument.C — insert a per-wave flush probe at kernel EXIT that streams a
 *  record out via a chosen egress helper, for the pass-by-address vs by-value evaluation.
 *
 *  Inserts  <probe>(pw.address(), nbytes)  at BPatch_exit, where pw is a Dyninst-managed
 *  per-wave variable (its address() is this wave's host-readable slice). Pick the probe:
 *    real_write -> gpu_real_fwrite (host reads the record by address; no 512B cap)
 *    bv_write   -> gpu_fwrite      (by-value copy into the ring; capped at 512B)
 *
 *  Usage: real_write_instrument <in.co> <out.co> [kernel] [lib] [probe] [nbytes]
 */
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "BPatch.h"
#include "BPatch_binaryEdit.h"
#include "BPatch_function.h"
#include "BPatch_point.h"
#include "BPatch_snippet.h"

using namespace Dyninst;
static BPatch bpatch;

static BPatch_function *find(BPatch_image *img, const char *name) {
  BPatch_Vector<BPatch_function *> fs;
  if (!img->findFunction(name, fs, true, false, /*incUninstrumentable=*/true) || fs.empty())
    return nullptr;
  return fs[0];
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <in.co> <out.co> [kernel] [lib] [probe] [nbytes]\n";
    return EXIT_FAILURE;
  }
  const char *in = argv[1], *out = argv[2];
  const char *kernelName = (argc > 3) ? argv[3] : "_Z9vectoraddPfPKfS1_i";
  const char *lib        = (argc > 4) ? argv[4] : "instrumentation/user_lib/combined.aliased.elf";
  const char *probeName  = (argc > 5) ? argv[5] : "real_write";
  const int   nbytes     = (argc > 6) ? atoi(argv[6]) : 2000;

  BPatch_binaryEdit *bin = bpatch.openBinary(in, /*openDependencies=*/true);
  if (!bin || !bin->loadLibrary(lib)) {
    std::cerr << "failed to open '" << in << "' or load '" << lib << "'\n";
    return EXIT_FAILURE;
  }
  BPatch_image *img = bin->getImage();
  BPatch_function *kernel = find(img, kernelName);
  BPatch_function *probe  = find(img, probeName);
  if (!kernel || !probe) {
    std::cerr << "missing kernel '" << kernelName << "' or probe '" << probeName << "'\n";
    return EXIT_FAILURE;
  }

  // Per-wave slice: filename at [0,128) + the nbytes record staged at slice+128.
  unsigned pwBytes = 128u + (unsigned)(nbytes > 0 ? nbytes : 0);
  BPatch_perWaveVar pw(pwBytes);
  bin->allocatePerWave(pwBytes);                        // arena-size the stride (was fixed 4096)
  int inserted = 0;
  if (auto *xpts = kernel->findPoint(BPatch_exit)) {
    BPatch_snippet base = pw.address();
    BPatch_constExpr nb(nbytes);
    BPatch_Vector<BPatch_snippet *> args{ &base, &nb };
    if (bin->insertSnippet(BPatch_funcCallExpr(*probe, args), *xpts, BPatch_callBefore, BPatch_lastSnippet))
      inserted = 1;
  }

  if (!bin->writeFile(out)) { std::cerr << "writeFile '" << out << "' failed\n"; return EXIT_FAILURE; }
  std::cout << "real_write: inserted " << probeName << "(pw.address(), " << nbytes
            << ") @exit (" << inserted << ") -> " << out << "\n";
  std::cout << "pw_stride=" << bin->perWaveStride() << "\n";   // harness bakes __dyninst_pw_stride
  return EXIT_SUCCESS;
}
