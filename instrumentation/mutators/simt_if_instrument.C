/*
 *  simt_if_instrument.C — MWE of SIMT-aware instrumentation ADDRESSING (SIMT-CFG → Project I).
 *
 *  Instead of instrumenting every basic block (bb_count), this targets the SEMANTIC SIMT-IF
 *  condition points: each `s_*_saveexec_b64` is the head of a SIMT-CONDITION block (the divergence
 *  point). We resolve each SimtIf's condition to its saveexec ADDRESS and insert a per-wave counter
 *  there via BPatch's instruction-level findPoint(addr) — so `seen[k]++` counts how many times each
 *  SIMT-IF condition #k is reached, per wave. That is the proposal's §3.1 `seen_count`, but driven by
 *  SIMT addressing rather than blanket per-block.
 *
 *  Addressing note: the condition address IS the saveexec head — the same point our ParseAPI
 *  SIMT-CFG (analysis/cfrecovery/simtcfg) resolves as SimtIf.condition(). The BPatch CFG here is not
 *  exec-split, but findPoint(addr) inserts exactly at that mid-block address, so no split is needed
 *  on the instrument side (address-based handoff; the analysis-side split is an analysis convenience).
 *
 *  Reuses the bb_inc / bb_flush_pw per-wave probes. getpc-free kernels only (see bb_count note).
 *  Usage: simt_if_instrument <in.co> <out.co> [kernel] [lib]
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
#include "Instruction.h"
#include "entryIDs.h"

using namespace Dyninst;
static BPatch bpatch;

// SIMT-CONDITION head: the saveexec family, by entryID (no string parsing) — the isModifyExecMask
// primitive inlined here so the mutator is self-contained.
// Arch dispatch (gfx908 + gfx940/gfx942): the SAVEEXEC family is identical GFX9 ISA, but the decoder
// emits DISTINCT per-arch entryID enumerators (gfx942 decodes via the gfx940 backend). Match both —
// one binary is one arch, so the extra labels never collide.
#define SIMT_OP(name) case amdgpu_gfx908_op_##name: case amdgpu_gfx940_op_##name
static bool isSaveExec(const InstructionAPI::Instruction &in) {
  switch (in.getOperation().getID()) {
    SIMT_OP(S_AND_SAVEEXEC_B64):  SIMT_OP(S_OR_SAVEEXEC_B64):
    SIMT_OP(S_XOR_SAVEEXEC_B64):  SIMT_OP(S_ANDN2_SAVEEXEC_B64):
    SIMT_OP(S_ORN2_SAVEEXEC_B64): SIMT_OP(S_NAND_SAVEEXEC_B64):
    SIMT_OP(S_NOR_SAVEEXEC_B64):  SIMT_OP(S_XNOR_SAVEEXEC_B64):
      return true;
    default: return false;
  }
}

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

  // --- SIMT addressing: the parser splits at every exec-writer (IA_amdgpu::isModifyExecMask), so a
  //     SIMT-IF CONDITION is now a real block whose FIRST instruction is a saveexec. We instrument its
  //     ENTRY (before the saveexec) — exactly a block-entry probe, no split or mid-block findPoint
  //     needed. Entry-before-saveexec is the proposal's §3.1 `seen_count` point. ---
  BPatch_flowGraph *cfg = kernel->getCFG();
  if (!cfg) { std::cerr << "getCFG() failed\n"; return EXIT_FAILURE; }
  std::set<BPatch_basicBlock *> bbset;
  cfg->getAllBasicBlocks(bbset);
  std::vector<BPatch_basicBlock *> blocks(bbset.begin(), bbset.end());
  std::sort(blocks.begin(), blocks.end(), [](BPatch_basicBlock *a, BPatch_basicBlock *b) {
    return a->getStartAddress() < b->getStartAddress();
  });
  std::vector<BPatch_basicBlock *> condBlocks;         // SimtIf #k CONDITION block (head == saveexec)
  for (BPatch_basicBlock *bb : blocks) {
    std::vector<std::pair<InstructionAPI::Instruction, Address>> insns;
    if (!bb->getInstructions(insns) || insns.empty()) continue;
    if (isSaveExec(insns.front().first))               // block HEAD is a saveexec -> SIMT-CONDITION
      condBlocks.push_back(bb);
  }
  const int nIf = (int)condBlocks.size();
  std::cout << "simt_if: " << nIf << " SIMT-IF condition block(s) in " << kernelName << "\n";
  for (int k = 0; k < nIf; k++)
    std::cout << "  SimtIf #" << k << " condition block @ 0x" << std::hex
              << condBlocks[k]->getStartAddress() << std::dec << "\n";
  if (nIf == 0) { std::cout << "simt_if: no divergence; nothing to instrument\n"; return EXIT_SUCCESS; }

  // Per-wave counters: seen[k] for each SIMT-IF, plus flush staging (mirrors bb_count).
  BPatch_perWaveVar counts(bin, (unsigned)nIf * 4u);
  BPatch_perWaveVar fname(bin, 64u);
  BPatch_perWaveVar report(bin, (unsigned)nIf * 24u + 16u);

  // --- insert seen[k]++ at each SIMT-IF CONDITION block ENTRY (before the saveexec) ---
  int inserted = 0;
  for (int k = 0; k < nIf; k++) {
    BPatch_point *pt = condBlocks[k]->findEntryPoint();
    if (!pt) { std::cerr << "  no entry point for condition block " << k << "\n"; continue; }
    BPatch_snippet base = counts.address();
    BPatch_constExpr id(k);
    BPatch_Vector<BPatch_snippet *> args{ &base, &id };
    if (bin->insertSnippet(BPatch_funcCallExpr(*inc, args), *pt, BPatch_callBefore, BPatch_lastSnippet))
      inserted++;
  }

  if (auto *xpts = kernel->findPoint(BPatch_exit)) {
    BPatch_snippet cAddr = counts.address(), fAddr = fname.address(), rAddr = report.address();
    BPatch_constExpr nC(nIf);
    BPatch_Vector<BPatch_snippet *> fargs{ &cAddr, &fAddr, &rAddr, &nC };
    bin->insertSnippet(BPatch_funcCallExpr(*flush, fargs), *xpts, BPatch_callBefore, BPatch_lastSnippet);
  }

  if (!bin->writeFile(out)) { std::cerr << "writeFile '" << out << "' failed\n"; return EXIT_FAILURE; }
  std::cout << "simt_if: instrumented " << inserted << "/" << nIf
            << " SIMT-IF condition point(s) with seen[k]++  -> " << out << "\n";
  std::cout << "pw_stride=" << bin->perWaveStride() << "\n";
  return EXIT_SUCCESS;
}
