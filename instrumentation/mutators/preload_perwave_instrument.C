/*
 *  preload_perwave_instrument.C — per-wave multi-file instrumentation for the
 *  LD_PRELOAD path. Inserts pw_open(pw.address()) @entry and pw_flush(pw.address())
 *  @exit. Each wave opens its own "wave_<wid>.txt" (via the gpu_fopen hostcall) and
 *  writes a line to it, keeping per-wave state in its slice of a Dyninst-managed per-wave
 *  variable (BPatch_perWaveVar) delivered by the launch-time kernarg PerWaveBuf — the
 *  preload allocates the buffer and appends it as the extra kernarg.
 *
 *  Usage: preload_perwave_instrument <in.co> <out.co> [kernel] [combined_lib]
 */
#include <cstdlib>
#include <iostream>
#include <vector>

#include "BPatch.h"
#include "BPatch_binaryEdit.h"
#include "BPatch_function.h"
#include "BPatch_point.h"
#include "BPatch_snippet.h"

using namespace Dyninst;
BPatch bpatch;

static BPatch_function *find(BPatch_image *img, const char *name) {
  BPatch_Vector<BPatch_function *> fs;
  if (!img->findFunction(name, fs, true, false, /*incUninstrumentable=*/true) || fs.empty())
    return nullptr;
  return fs[0];
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <in.co> <out.co> [kernel] [combined_lib]\n";
    return EXIT_FAILURE;
  }
  const char *in = argv[1], *out = argv[2];
  const char *kernelName = (argc > 3) ? argv[3] : "_Z9vectoraddPfPKfS1_i";
  const char *lib        = (argc > 4) ? argv[4] : "user_lib/combined.aliased.elf";

  BPatch_binaryEdit *bin = bpatch.openBinary(in, /*openDependencies=*/true);
  if (!bin || !bin->loadLibrary(lib)) {
    std::cerr << "failed to open '" << in << "' or load '" << lib << "'\n";
    return EXIT_FAILURE;
  }
  BPatch_image *img = bin->getImage();

  BPatch_function *pwOpen  = find(img, "pw_open");
  BPatch_function *pwFlush = find(img, "pw_flush");
  BPatch_function *kernel  = find(img, kernelName);
  if (!pwOpen || !pwFlush || !kernel) {
    std::cerr << "missing pw_open/pw_flush or kernel '" << kernelName << "'\n";
    return EXIT_FAILURE;
  }

  // Dyninst-managed per-wave variable; address() is THIS wave's slice base (kernarg PerWaveBuf).
  BPatch_perWaveVar pw(/*bytesPerWave=*/4096);

  if (auto *e = kernel->findPoint(BPatch_entry)) {
    BPatch_snippet base = pw.address();                 // this wave's slice pointer
    BPatch_Vector<BPatch_snippet *> args{ &base };
    bin->insertSnippet(BPatch_funcCallExpr(*pwOpen, args), *e, BPatch_callBefore, BPatch_lastSnippet);
  }
  if (auto *x = kernel->findPoint(BPatch_exit)) {
    BPatch_snippet base = pw.address();
    BPatch_Vector<BPatch_snippet *> args{ &base };
    bin->insertSnippet(BPatch_funcCallExpr(*pwFlush, args), *x, BPatch_callBefore, BPatch_lastSnippet);
  }

  if (!bin->writeFile(out)) {
    std::cerr << "failed to write '" << out << "'\n";
    return EXIT_FAILURE;
  }
  std::cout << "wrote " << out << " (pw_open(pw.address()) @entry, pw_flush(pw.address()) @exit)\n";
  return EXIT_SUCCESS;
}
