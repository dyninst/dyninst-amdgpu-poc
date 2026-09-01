// slice_smoke.C — smoke-test the DataflowAPI Slicer/Absloc path on an AMDGPU (gfx908)
// code object. Finds a v_cndmask, backward-slices from its defining assignment, and
// checks the slice reaches the v_cmp that produced its predicate mask.
//
// Confirms Layer-1 (def-use slicing via the generic getReadSet/getWriteSet path) works
// for arbitrary AMDGPU opcodes — the prerequisite for mask lineage + condition provenance.
#include <cstdio>
#include <string>
#include <map>
#include "CodeObject.h"
#include "CodeSource.h"
#include "slicing.h"
#include "AbslocInterface.h"
#include "Absloc.h"
#include "Instruction.h"
#include "Graph.h"

using namespace Dyninst;
using namespace Dyninst::ParseAPI;
using namespace Dyninst::InstructionAPI;

static bool has(const std::string &s, const char *sub) { return s.find(sub) != std::string::npos; }

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <code-object> [func-substr]\n", argv[0]); return 2; }
  const char *fname = argv[1];
  std::string want = (argc >= 3) ? argv[2] : "";

  SymtabCodeSource *scs = new SymtabCodeSource((char*)fname);
  CodeObject *co = new CodeObject(scs);
  co->parse();
  printf("parsed %s : %zu functions\n", fname, co->funcs().size());

  AssignmentConverter ac(/*cache=*/true, /*stackAnalysis=*/false);

  for (auto *f : co->funcs()) {
    if (!want.empty() && !has(f->name(), want.c_str())) continue;

    // Build addr -> mnemonic string for this function (for labeling slice nodes).
    std::map<Address,std::string> insnStr;
    Block *cndBlock = nullptr; Address cndAddr = 0; Instruction cndInsn;
    for (auto *b : f->blocks()) {
      Block::Insns insns; b->getInsns(insns);
      for (auto &kv : insns) {
        std::string s = kv.second.format();
        insnStr[kv.first] = s;
        if (!cndBlock && has(s, "V_CNDMASK")) { cndBlock = b; cndAddr = kv.first; cndInsn = kv.second; }
      }
    }
    if (!cndBlock) {
      if (getenv("SLICE_DUMP")) {
        printf("  [dump] function '%s' instruction formats:\n", f->name().c_str());
        for (auto &kv : insnStr) printf("    0x%lx  %s\n", (unsigned long)kv.first, kv.second.c_str());
      }
      continue;   // no predication in this function; try next
    }

    printf("\n=== function '%s' : slicing from v_cndmask @ 0x%lx ===\n  target: %s\n",
           f->name().c_str(), (unsigned long)cndAddr, insnStr[cndAddr].c_str());

    // Convert the v_cndmask into assignments; pick one with a defined output.
    std::vector<Assignment::Ptr> assigns;
    ac.convert(cndInsn, cndAddr, f, cndBlock, assigns);
    printf("  AssignmentConverter produced %zu assignment(s)\n", assigns.size());
    Assignment::Ptr target;
    for (auto &a : assigns) { printf("    assign: %s\n", a->format().c_str()); if (!target) target = a; }
    if (!target) { printf("  NO assignment produced -> Absloc path did NOT crack this opcode\n"); return 1; }

    // Backward slice from that assignment with default predicates.
    Slicer slicer(target, cndBlock, f, /*cache=*/false, /*stackAnalysis=*/false);
    Slicer::Predicates preds;
    GraphPtr g = slicer.backwardSlice(preds);
    if (!g) { printf("  backwardSlice returned NULL\n"); return 1; }

    // Enumerate slice nodes; report reach to a v_cmp.
    NodeIterator nb, ne; g->allNodes(nb, ne);
    int n = 0, vcmp = 0, vcmpx = 0;
    printf("  --- slice nodes ---\n");
    for (; nb != ne; ++nb) {
      SliceNode::Ptr sn = boost::dynamic_pointer_cast<SliceNode>(*nb);
      if (!sn || !sn->assign()) continue;
      n++;
      Address a = sn->assign()->addr();
      std::string s = insnStr.count(a) ? insnStr[a] : std::string("<?>");
      printf("    [%2d] 0x%lx  %s\n", n, (unsigned long)a, s.c_str());
      if (has(s, "V_CMP")) vcmp++;
      if (has(s, "V_CMPX")) vcmpx++;
    }
    printf("  --- %d slice nodes; reached v_cmp: %s (%d) ---\n", n, vcmp ? "YES" : "no", vcmp);
    printf("\nRESULT: Slicer built a %d-node backward slice on AMDGPU without crashing; "
           "predicate provenance (v_cmp) %s.\n", n, vcmp ? "REACHED" : "not reached in this func");
    return vcmp ? 0 : 3;   // 0 = fully confirmed; 3 = slice built but didn't reach v_cmp
  }
  printf("no v_cndmask found in the selected function(s)\n");
  return 4;
}
