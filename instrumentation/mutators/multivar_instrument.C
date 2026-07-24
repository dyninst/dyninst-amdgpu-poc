/*
 *  multivar_instrument.C — MULTIPLE per-wave variables at distinct arena offsets.
 *
 *  Declares two per-wave variables (v0, v1) via the arena bump-allocator, then inserts
 *  pw_mark2(v0.address(), v1.address()) at kernel exit. Each address() carries its own
 *  byte offset within the wave's slice, so emitCall lowers base + wid*STRIDE + offset per
 *  argument (the per-arg VGPR offset-add). The probe writes a distinct marker to each; the
 *  launcher's per-wave slice dump then shows the two markers at different words, proving the
 *  offsets landed correctly.
 *
 *  Usage: multivar_instrument <in.co> <out.co> [kernel] [lib]
 */
#include <cstdlib>
#include <iostream>

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
    std::cerr << "usage: " << argv[0] << " <in.co> <out.co> [kernel] [lib]\n";
    return EXIT_FAILURE;
  }
  const char *in = argv[1], *out = argv[2];
  const char *kernelName = (argc > 3) ? argv[3] : "_Z9vectoraddPfPKfS1_i";
  const char *lib        = (argc > 4) ? argv[4] : "instrumentation/user_lib/combined.aliased.elf";

  BPatch_binaryEdit *bin = bpatch.openBinary(in, /*openDependencies=*/true);
  if (!bin || !bin->loadLibrary(lib)) {
    std::cerr << "failed to open '" << in << "' or load '" << lib << "'\n";
    return EXIT_FAILURE;
  }
  BPatch_image *img = bin->getImage();
  BPatch_function *kernel = find(img, kernelName);
  BPatch_function *mark2  = find(img, "pw_mark2");
  if (!kernel || !mark2) {
    std::cerr << "missing kernel '" << kernelName << "' or pw_mark2 in lib\n";
    return EXIT_FAILURE;
  }

  // Two per-wave variables at distinct arena offsets (v0 @0, v1 @8 after 8-alignment).
  unsigned o0 = bin->allocatePerWave(4);
  unsigned o1 = bin->allocatePerWave(4);
  BPatch_perWaveVar v0(4, o0), v1(4, o1);

  int inserted = 0;
  if (auto *xpts = kernel->findPoint(BPatch_exit)) {
    BPatch_snippet a = v0.address(), b = v1.address();     // base+wid*STRIDE+o0 / +o1
    BPatch_Vector<BPatch_snippet *> args{ &a, &b };
    if (bin->insertSnippet(BPatch_funcCallExpr(*mark2, args), *xpts, BPatch_callBefore, BPatch_lastSnippet))
      inserted = 1;
  }

  if (!bin->writeFile(out)) { std::cerr << "writeFile '" << out << "' failed\n"; return EXIT_FAILURE; }
  std::cout << "multivar: v0@" << o0 << " v1@" << o1 << ", inserted pw_mark2(v0,v1) @exit ("
            << inserted << ") -> " << out << "\n";
  std::cout << "pw_stride=" << bin->perWaveStride() << "\n";   // harness bakes __dyninst_pw_stride
  return EXIT_SUCCESS;
}
