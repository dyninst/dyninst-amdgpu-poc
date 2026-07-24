/*
 *  bb_count_multivar_instrument.C — basic-block execution counter, ONE per-wave variable
 *  PER BASIC BLOCK (the multi-var style).
 *
 *  Instead of a single counts[] blob indexed by bbid (bb_count_instrument.C), this declares
 *  nbb separate 4-byte per-wave counters — densely packed by the arena at slice offsets
 *  0,4,8,... — and inserts bb_inc_one(ctr[k].address()) at each block entry. The probe just
 *  bumps the cell it is handed (no bbid, no indexing); Dyninst bakes each block's offset via
 *  the per-arg add. bb_flush_pw(ctr[0].address(), nbb) @exit walks the contiguous counters,
 *  so the on-disk output is identical to bb_count_instrument.
 *
 *  This is a style choice for the end user: framework-owned per-block variables vs. one
 *  indexed array. getpc-free kernels only.
 *
 *  Usage: bb_count_multivar_instrument <in.co> <out.co> [kernel] [lib]
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
  const char *kernelName = (argc > 3) ? argv[3] : "_Z6bbdemoPiPKii";
  const char *lib        = (argc > 4) ? argv[4] : "instrumentation/user_lib/combined.aliased.elf";

  BPatch_binaryEdit *bin = bpatch.openBinary(in, /*openDependencies=*/true);
  if (!bin || !bin->loadLibrary(lib)) {
    std::cerr << "failed to open '" << in << "' or load '" << lib << "'\n";
    return EXIT_FAILURE;
  }
  BPatch_image *img = bin->getImage();
  BPatch_function *kernel = find(img, kernelName);
  BPatch_function *inc    = find(img, "bb_inc_one");
  BPatch_function *flush  = find(img, "bb_flush_pw");
  if (!kernel || !inc || !flush) {
    std::cerr << "missing kernel '" << kernelName << "' or bb_inc_one/bb_flush_pw in lib\n";
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

  // One 4-byte per-wave counter per block, densely packed (4-aligned) => a contiguous
  // counts[] at offsets 0,4,...,(nbb-1)*4. Then reserve the flush scratch right after
  // (8-aligned, which equals bb_flush_pw's C = align8(nbb*4)) so the STRIDE covers it and
  // bb_flush_pw reads the counters + formats the report exactly as in bb_count_instrument.
  std::vector<BPatch_perWaveVar> ctr;
  for (int k = 0; k < nbb; k++) ctr.emplace_back(4, bin->allocatePerWave(4, /*align=*/4));
  bin->allocatePerWave(64u + (unsigned)nbb * 20u + 16u, /*align=*/8);   // flush scratch

  int inserted = 0;
  for (int k = 0; k < nbb; k++) {
    BPatch_point *p = v[k]->findEntryPoint();
    if (!p) continue;
    BPatch_snippet c = ctr[k].address();                 // THIS block's own counter cell
    BPatch_Vector<BPatch_snippet *> args{ &c };
    if (bin->insertSnippet(BPatch_funcCallExpr(*inc, args), *p, BPatch_callBefore, BPatch_lastSnippet))
      inserted++;
  }

  if (auto *xpts = kernel->findPoint(BPatch_exit)) {
    BPatch_perWaveVar base0(4, 0);                        // slice base (offset 0)
    BPatch_snippet fbase = base0.address();
    BPatch_constExpr nbbC(nbb);
    BPatch_Vector<BPatch_snippet *> fargs{ &fbase, &nbbC };
    bin->insertSnippet(BPatch_funcCallExpr(*flush, fargs), *xpts, BPatch_callBefore, BPatch_lastSnippet);
  }

  if (!bin->writeFile(out)) { std::cerr << "writeFile '" << out << "' failed\n"; return EXIT_FAILURE; }
  std::cout << "bb_count_multivar: " << nbb << " blocks, inserted " << inserted
            << " bb_inc_one(ctr[k].address()) + bb_flush_pw@exit -> " << out << "\n";
  std::cout << "pw_stride=" << bin->perWaveStride() << "\n";
  return EXIT_SUCCESS;
}
