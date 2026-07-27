/*
 *  real_write_instrument.C — insert a per-wave flush probe at kernel EXIT that streams a
 *  record out via a chosen egress helper, for the pass-by-address vs by-value evaluation.
 *
 *  Inserts  <probe>(fname.address(), record.address(), nbytes)  at BPatch_exit, where
 *  fname/record are two Dyninst-managed per-wave variables (each address() is this wave's
 *  host-readable staging buffer). Pick the probe:
 *    real_write -> gpu_fwrite     pass-by-address (host reads the record by address; no 512B cap)
 *    bv_write   -> gpu_fwrite_256 by-value copy into the ring; capped at 512B
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

  // Two distinct per-wave variables (NOT one slice the probe sub-divides): a filename
  // staging buffer and the nbytes record buffer. The arena packs them and sizes the stride.
  BPatch_perWaveVar fname (bin, 64u);                                   // filename staging
  BPatch_perWaveVar record(bin, (unsigned)(nbytes > 0 ? nbytes : 0));   // record buffer
  int inserted = 0;
  if (auto *xpts = kernel->findPoint(BPatch_exit)) {
    BPatch_snippet fAddr = fname.address(), rAddr = record.address();
    BPatch_constExpr nb(nbytes);
    BPatch_Vector<BPatch_snippet *> args{ &fAddr, &rAddr, &nb };
    if (bin->insertSnippet(BPatch_funcCallExpr(*probe, args), *xpts, BPatch_callBefore, BPatch_lastSnippet))
      inserted = 1;
  }

  if (!bin->writeFile(out)) { std::cerr << "writeFile '" << out << "' failed\n"; return EXIT_FAILURE; }
  std::cout << "real_write: inserted " << probeName << "(fname, record, " << nbytes
            << ") @exit (" << inserted << ") -> " << out << "\n";
  std::cout << "pw_stride=" << bin->perWaveStride() << "\n";   // harness bakes __dyninst_pw_stride
  return EXIT_SUCCESS;
}
