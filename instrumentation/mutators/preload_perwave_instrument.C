/*
 *  preload_perwave_instrument.C — per-wave multi-file instrumentation for the
 *  LD_PRELOAD path. Inserts pw_open(...) @entry and pw_flush(...) @exit. Each wave opens
 *  its own "wave_<wid>.txt" (via the gpu_fopen hostcall) and writes a line to it.
 *
 *  This is the SHOWCASE for named per-wave variables: rather than handing the probes one
 *  slice they sub-divide by hand, the mutator declares FIVE per-wave variables and passes
 *  each probe just the ones it needs. Three are persistent state carried from entry to
 *  exit — handle (this wave's open file), hits, lanes — and two are transient staging —
 *  name (the filename) and line (the output text). All are BPatch_perWaveVars delivered by
 *  the launch-time kernarg PerWaveBuf; the preload allocates the buffer and appends it as
 *  the extra kernarg.
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

  // Named per-wave variables (each address() is THIS wave's own cell, delivered via the
  // kernarg PerWaveBuf). handle/hits/lanes are shared between the two probes (persistent
  // per-wave state); name/line are each probe's private staging buffer. The arena packs
  // them and sizes the stride.
  BPatch_perWaveVar handle(bin, 8u);                    // this wave's open file handle
  BPatch_perWaveVar hits  (bin, 4u);                    // per-wave accumulator
  BPatch_perWaveVar lanes (bin, 4u);                    // active-lane count at last hit
  BPatch_perWaveVar name  (bin, 64u);                   // filename staging (pw_open)
  BPatch_perWaveVar line  (bin, 128u);                  // output-line staging (pw_flush)

  if (auto *e = kernel->findPoint(BPatch_entry)) {
    BPatch_snippet hAddr = handle.address(), hiAddr = hits.address(),
                   lAddr = lanes.address(), nAddr = name.address();
    BPatch_Vector<BPatch_snippet *> args{ &hAddr, &hiAddr, &lAddr, &nAddr };
    bin->insertSnippet(BPatch_funcCallExpr(*pwOpen, args), *e, BPatch_callBefore, BPatch_lastSnippet);
  }
  if (auto *x = kernel->findPoint(BPatch_exit)) {
    BPatch_snippet hAddr = handle.address(), hiAddr = hits.address(),
                   lAddr = lanes.address(), tAddr = line.address();
    BPatch_Vector<BPatch_snippet *> args{ &hAddr, &hiAddr, &lAddr, &tAddr };
    bin->insertSnippet(BPatch_funcCallExpr(*pwFlush, args), *x, BPatch_callBefore, BPatch_lastSnippet);
  }

  if (!bin->writeFile(out)) {
    std::cerr << "failed to write '" << out << "'\n";
    return EXIT_FAILURE;
  }
  std::cout << "wrote " << out << " (pw_open(handle,hits,lanes,name) @entry, "
               "pw_flush(handle,hits,lanes,line) @exit)\n";
  std::cout << "pw_stride=" << bin->perWaveStride() << "\n";   // harness bakes __dyninst_pw_stride
  return EXIT_SUCCESS;
}
