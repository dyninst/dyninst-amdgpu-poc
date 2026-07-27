/*
 *  bb_count_instrument.C — basic-block execution counter (getpc-free kernels).
 *
 *  Enumerates the kernel's basic blocks and inserts `bb_inc(counts.address(), bbid)` at
 *  every block entry plus `bb_flush_pw(counts, fname, report, nbb)` at kernel exit. The
 *  mutator declares THREE distinct per-wave variables — the u32 counts array, a filename
 *  staging buffer, and the report-text staging buffer — rather than sub-dividing one blob;
 *  the arena packs them and the probe does no offset math. Each is delivered by the
 *  launch-time kernarg PerWaveBuf; each wave accumulates per-block counts in its slice and
 *  bb_flush_pw writes bbcount_<wid>.txt. Run under the LD_PRELOAD path (the preload
 *  allocates the per-wave buffer + appends it as the extra kernarg).
 *
 *  NOTE: intended for getpc-free kernels (no calls / no PC-relative data). Relocating a
 *  getpc idiom mid-block is not yet supported (see docs/ROADMAP.md "Current frontier").
 *  Check the target with: llvm-objdump -d <co> | grep s_getpc  (expect none).
 *
 *  Usage: bb_count_instrument <in.co> <out.co> [kernel] [lib]
 */
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "BPatch.h"
#include "BPatch_binaryEdit.h"
#include "BPatch_function.h"
#include "BPatch_flowGraph.h"
#include "BPatch_basicBlock.h"
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
  const char *kernelName = (argc > 3) ? argv[3] : "_Z4vaddPfPKfS1_i";
  const char *lib = (argc > 4) ? argv[4] : "instrumentation/user_lib/combined.aliased.elf";

  BPatch_binaryEdit *bin = bpatch.openBinary(in, /*openDependencies=*/true);
  if (!bin || !bin->loadLibrary(lib)) {
    std::cerr << "failed to open '" << in << "' or load '" << lib << "'\n";
    return EXIT_FAILURE;
  }
  BPatch_image *img = bin->getImage();
  BPatch_function *kernel = find(img, kernelName);
  BPatch_function *inc = find(img, "bb_inc");
  BPatch_function *flush = find(img, "bb_flush_pw");
  if (!kernel || !inc || !flush) {
    std::cerr << "missing kernel '" << kernelName << "' or bb_inc/bb_flush_pw in lib\n";
    return EXIT_FAILURE;
  }

  BPatch_flowGraph *cfg = kernel->getCFG();
  if (!cfg) { std::cerr << "getCFG() failed\n"; return EXIT_FAILURE; }
  std::set<BPatch_basicBlock *> bbs;
  cfg->getAllBasicBlocks(bbs);
  std::vector<BPatch_basicBlock *> v(bbs.begin(), bbs.end());
  std::sort(v.begin(), v.end(), [](BPatch_basicBlock *a, BPatch_basicBlock *b) {
    return a->getStartAddress() < b->getStartAddress();
  });
  const int nbb = (int)v.size();

  // Declare the buffers the probes need as SEPARATE per-wave variables (NOT one blob the
  // probe sub-divides): the u32 counts[nbb] that bb_inc bumps, a 64B filename staging
  // buffer, and the report text (one "bb <k>: <count>\n" line per block, ~20B each + slack).
  // The arena packs them densely and reports the STRIDE (rounded to the emitter's stride,
  // baked into the co as __dyninst_pw_stride); bb_flush_pw is handed each address directly.
  unsigned nbbu = (unsigned)(nbb > 0 ? nbb : 0);
  BPatch_perWaveVar counts(bin, nbbu * 4u);            // u32 counts[nbb]  (bb_inc target)
  BPatch_perWaveVar fname (bin, 64u);                  // filename staging
  BPatch_perWaveVar report(bin, nbbu * 20u + 16u);     // report text (host-readable for fwrite)

  int inserted = 0;
  for (int idx = 0; idx < nbb; idx++) {
    BPatch_point *p = v[idx]->findEntryPoint();
    if (!p) continue;
    BPatch_snippet base = counts.address();              // this wave's counts array
    BPatch_constExpr bbid(idx);
    BPatch_Vector<BPatch_snippet *> args{ &base, &bbid };
    if (bin->insertSnippet(BPatch_funcCallExpr(*inc, args), *p, BPatch_callBefore, BPatch_lastSnippet))
      inserted++;
  }

  if (auto *xpts = kernel->findPoint(BPatch_exit)) {   // findPoint returns a vector of points
    BPatch_snippet cAddr = counts.address(), fAddr = fname.address(), rAddr = report.address();
    BPatch_constExpr nbbC(nbb);
    BPatch_Vector<BPatch_snippet *> fargs{ &cAddr, &fAddr, &rAddr, &nbbC };
    bin->insertSnippet(BPatch_funcCallExpr(*flush, fargs), *xpts, BPatch_callBefore, BPatch_lastSnippet);
  }

  if (!bin->writeFile(out)) { std::cerr << "writeFile '" << out << "' failed\n"; return EXIT_FAILURE; }
  std::cout << "bb_count: " << nbb << " blocks, inserted " << inserted
            << " bb_inc(counts,bbid) probe(s) + bb_flush_pw(counts,fname,report,nbb)@exit -> " << out << "\n";
  // Machine-readable line the harness parses to inject __dyninst_pw_stride into the co.
  std::cout << "pw_stride=" << bin->perWaveStride() << "\n";
  return EXIT_SUCCESS;
}
