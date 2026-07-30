/*
 *  getpc_reloc_instrument.C — validation mutator for AMDGPU getpc-call RELOCATION.
 *
 *  Instruments a kernel that contains the compiler's getpc+add(+addc)->swappc call
 *  idiom (see mutatees/vaddcall.cpp). Instrumenting relocates the kernel's blocks, so
 *  every original s_getpc lands at a new address; its baked add/addc offset must be
 *  corrected at emit time (PCWidget-amdgpu.C PCtoReg + IPPatch::apply). This mutator
 *  just needs to FORCE that relocation and leave the kernel otherwise computing the
 *  same result, so the run itself (out == A+B, no fault) is the correctness signal.
 *
 *  It also scans the kernel and reports every getpc / swappc site (so the harness can
 *  confirm the idiom is present), and inserts a hostcall probe both at entry and, best
 *  effort, right at the first getpc site (to guarantee that specific block is split and
 *  relocated).
 *
 *  Usage: getpc_reloc_instrument <in.co> <out.co> <kernel> <hostcall_lib>
 */
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "BPatch.h"
#include "BPatch_binaryEdit.h"
#include "BPatch_basicBlock.h"
#include "BPatch_flowGraph.h"
#include "BPatch_function.h"
#include "BPatch_point.h"
#include "BPatch_snippet.h"
#include "Instruction.h"

using namespace Dyninst;
BPatch bpatch;

static BPatch_function *find(BPatch_image *img, const char *name) {
  BPatch_Vector<BPatch_function *> fs;
  if (!img->findFunction(name, fs, true, false, /*incUninstrumentable=*/true) || fs.empty())
    return nullptr;
  return fs[0];
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: " << argv[0] << " <in.co> <out.co> <kernel> <hostcall_lib>\n";
    return EXIT_FAILURE;
  }
  const char *in = argv[1], *out = argv[2], *kernelName = argv[3], *lib = argv[4];

  BPatch_binaryEdit *bin = bpatch.openBinary(in, /*openDependencies=*/true);
  if (!bin || !bin->loadLibrary(lib)) { std::cerr << "open/load failed\n"; return EXIT_FAILURE; }
  BPatch_image *img = bin->getImage();

  BPatch_function *hcOpen  = find(img, "hc_open");
  BPatch_function *hcWrite = find(img, "hc_write_id");
  BPatch_function *hcClose = find(img, "hc_close");
  BPatch_function *kernel  = find(img, kernelName);
  if (!hcOpen || !hcWrite || !hcClose || !kernel) {
    std::cerr << "missing hc_* hooks or kernel '" << kernelName << "'\n"; return EXIT_FAILURE;
  }

  // Scan the kernel for the getpc idiom and report every getpc / swappc site.
  BPatch_flowGraph *cfg = kernel->getCFG();
  std::set<BPatch_basicBlock *> blockSet;
  std::vector<Address> getpcSites, swappcSites;
  if (cfg && cfg->getAllBasicBlocks(blockSet)) {
    std::vector<BPatch_basicBlock *> blocks(blockSet.begin(), blockSet.end());
    std::sort(blocks.begin(), blocks.end(), [](BPatch_basicBlock *a, BPatch_basicBlock *b) {
      return a->getStartAddress() < b->getStartAddress();
    });
    for (BPatch_basicBlock *bb : blocks) {
      std::vector<std::pair<InstructionAPI::Instruction, Address>> insns;
      if (!bb->getInstructions(insns)) continue;
      for (auto &ia : insns) {
        std::string m = ia.first.format();
        if (m.find("GETPC") != std::string::npos)  getpcSites.push_back(ia.second);
        if (m.find("SWAPPC") != std::string::npos) swappcSites.push_back(ia.second);
      }
    }
  }
  std::cout << "getpc_reloc: found " << getpcSites.size() << " s_getpc site(s), "
            << swappcSites.size() << " s_swappc site(s) in " << kernelName << "\n";
  for (Address a : getpcSites)  std::cout << "  getpc  @ 0x" << std::hex << a << std::dec << "\n";
  for (Address a : swappcSites) std::cout << "  swappc @ 0x" << std::hex << a << std::dec << "\n";
  if (getpcSites.empty())
    std::cout << "getpc_reloc: WARNING no getpc idiom found — relocation test would be vacuous\n";

  // Bracket the kernel: hc_open + hc_write_id(0) @entry, hc_close @exit. On AMDGPU this
  // relocates the whole function (incl. the getpc block).
  BPatch_Vector<BPatch_snippet *> noArgs;
  BPatch_constExpr id0(0);
  BPatch_Vector<BPatch_snippet *> idArg{&id0};
  if (auto *e = kernel->findPoint(BPatch_entry)) {
    bin->insertSnippet(BPatch_funcCallExpr(*hcOpen, noArgs), *e, BPatch_callBefore, BPatch_lastSnippet);
    bin->insertSnippet(BPatch_funcCallExpr(*hcWrite, idArg), *e, BPatch_callBefore, BPatch_lastSnippet);
  }
  if (auto *x = kernel->findPoint(BPatch_exit))
    bin->insertSnippet(BPatch_funcCallExpr(*hcClose, noArgs), *x, BPatch_callBefore, BPatch_lastSnippet);

  // Best effort: also insert a probe right at the first getpc site so that specific
  // mid-block point is split and relocated (belt-and-suspenders; entry alone already
  // relocates the function). OFF by default: inserting INSIDE the getpc->swappc block
  // splits the idiom and (with the current no-op CFWidget indirect-call handling) drops
  // the swappc. Enable with GETPC_PROBE_AT_SITE=1 to reproduce that failure mode.
  if (!getpcSites.empty() && getenv("GETPC_PROBE_AT_SITE")) {
    if (BPatch_point *pt = kernel->findPoint(getpcSites.front())) {
      BPatch_constExpr id1(1);
      BPatch_Vector<BPatch_snippet *> idArg1{&id1};
      if (bin->insertSnippet(BPatch_funcCallExpr(*hcWrite, idArg1), *pt, BPatch_callBefore, BPatch_lastSnippet))
        std::cout << "getpc_reloc: inserted probe at getpc site 0x" << std::hex
                  << getpcSites.front() << std::dec << " (forces its block to relocate)\n";
      else
        std::cout << "getpc_reloc: could not instrument the getpc site directly "
                     "(entry instrumentation still relocates it)\n";
    }
  }

  if (!bin->writeFile(out)) { std::cerr << "write failed\n"; return EXIT_FAILURE; }
  std::cout << "wrote " << out << "\n";
  return EXIT_SUCCESS;
}
