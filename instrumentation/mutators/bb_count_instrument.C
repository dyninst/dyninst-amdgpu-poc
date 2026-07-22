/*
 *  bb_count_instrument.C — basic-block execution counter (getpc-free kernels).
 *
 *  Enumerates the kernel's basic blocks and inserts `bb_hit(bbid)` at every block
 *  entry plus `bb_flush(nbb)` at kernel exit. Each wave accumulates per-block counts
 *  in its slice of the global per-wave buffer (g_pw_base); bb_flush writes
 *  bbcount_<wid>.txt. Run under the LD_PRELOAD path with HOSTCALL_PW_BYTES set so the
 *  preload allocates + defines g_pw_base.
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
  BPatch_function *hit = find(img, "bb_hit");
  BPatch_function *flush = find(img, "bb_flush");
  if (!kernel || !hit || !flush) {
    std::cerr << "missing kernel '" << kernelName << "' or bb_hit/bb_flush in lib\n";
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

  int inserted = 0;
  for (int idx = 0; idx < nbb; idx++) {
    BPatch_point *p = v[idx]->findEntryPoint();
    if (!p) continue;
    BPatch_Vector<BPatch_snippet *> args;
    args.push_back(new BPatch_constExpr(idx));           // bbid, as an immediate
    BPatch_funcCallExpr call(*hit, args);
    if (bin->insertSnippet(call, *p, BPatch_callBefore, BPatch_lastSnippet))
      inserted++;
  }

  if (auto *xpts = kernel->findPoint(BPatch_exit)) {   // findPoint returns a vector of points
    BPatch_Vector<BPatch_snippet *> fargs;
    fargs.push_back(new BPatch_constExpr(nbb));
    BPatch_funcCallExpr fcall(*flush, fargs);
    bin->insertSnippet(fcall, *xpts, BPatch_callBefore, BPatch_lastSnippet);
  }

  if (!bin->writeFile(out)) { std::cerr << "writeFile '" << out << "' failed\n"; return EXIT_FAILURE; }
  std::cout << "bb_count: " << nbb << " blocks, inserted " << inserted
            << " bb_hit probe(s) + bb_flush@exit -> " << out << "\n";
  return EXIT_SUCCESS;
}
