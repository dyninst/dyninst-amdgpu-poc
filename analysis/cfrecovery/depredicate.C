// depredicate.C — de-predication MVP. For each V_CNDMASK in a function, recover the
// symbolic CONDITION by backward-walking ONLY its mask operand's provenance (not the
// data operands), recognizing the v_cmp predicate leaf + the s_and/or/xor mask algebra.
// Sidesteps the missing SymEval AMDGPU semantics with a small purpose-built evaluator.
// Prints each select as `dst = cond ? T : F`, exposing the reconstructed if/else.
//
// Scope (MVP): reaching def = nearest prior writer by address (exact for branchless
// predicated regions — the de-predication target; approximate across branches). Operands
// read from InstructionAPI format() (uppercase). Multi-block-precise reaching-defs = TODO.
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "CodeObject.h"
#include "CodeSource.h"
#include "Instruction.h"

using namespace Dyninst;
using namespace Dyninst::ParseAPI;
using namespace Dyninst::InstructionAPI;

struct Insn { Address addr; std::string mnem; std::vector<std::string> ops; };

static std::vector<std::string> splitOps(const std::string &rest) {
  std::vector<std::string> v; std::string cur; int depth = 0;
  for (char c : rest) {
    if (c == '[') depth++; else if (c == ']') depth--;
    if (c == ',' && depth == 0) { // split on top-level commas
      while (!cur.empty() && cur.front() == ' ') cur.erase(cur.begin());
      while (!cur.empty() && cur.back()  == ' ') cur.pop_back();
      if (!cur.empty()) v.push_back(cur); cur.clear();
    } else cur += c;
  }
  while (!cur.empty() && cur.front() == ' ') cur.erase(cur.begin());
  while (!cur.empty() && cur.back()  == ' ') cur.pop_back();
  if (!cur.empty()) v.push_back(cur);
  return v;
}

static Insn parseInsn(Address a, const std::string &fmt) {
  Insn in; in.addr = a;
  size_t sp = fmt.find(' ');
  if (sp == std::string::npos) { in.mnem = fmt; return in; }
  in.mnem = fmt.substr(0, sp);
  in.ops = splitOps(fmt.substr(sp + 1));
  return in;
}

// V_CMP_<COND>_<TYPE> -> infix operator
static std::string cmpOp(const std::string &mnem) {
  std::vector<std::string> p; std::stringstream ss(mnem); std::string t;
  while (std::getline(ss, t, '_')) p.push_back(t);
  std::string c = p.size() > 2 ? p[2] : "?";
  if (c=="LT") return "<"; if (c=="GT") return ">"; if (c=="LE"||c=="NGT") return "<=";
  if (c=="GE"||c=="NLT") return ">="; if (c=="EQ") return "=="; if (c=="NE"||c=="LG"||c=="NEQ") return "!=";
  return "." + c + ".";
}

struct Ctx { std::vector<Insn> *ins; std::map<Address,int> *idxAt; };

// Nearest prior writer of register token `reg` before instruction index `before`.
static int defOf(Ctx &cx, const std::string &reg, int before) {
  for (int j = before - 1; j >= 0; --j)
    if (!(*cx.ins)[j].ops.empty() && (*cx.ins)[j].ops[0] == reg) return j;
  return -1;
}

static std::string buildCond(Ctx &cx, const std::string &reg, int before, int depth) {
  if (depth > 32) return reg + "…";
  int j = defOf(cx, reg, before);
  if (j < 0) return reg + "@in";                       // comes from outside the region (ABI / another block)
  const Insn &d = (*cx.ins)[j];
  const std::string &m = d.mnem;
  auto src = [&](int i){ return i < (int)d.ops.size() ? d.ops[i] : std::string("?"); };
  if (m.rfind("V_CMP", 0) == 0)   return "(" + src(1) + " " + cmpOp(m) + " " + src(2) + ")";
  if (m == "S_AND_B64")           return "(" + buildCond(cx,src(1),j,depth+1) + " & "  + buildCond(cx,src(2),j,depth+1) + ")";
  if (m == "S_OR_B64")            return "(" + buildCond(cx,src(1),j,depth+1) + " | "  + buildCond(cx,src(2),j,depth+1) + ")";
  if (m == "S_XOR_B64")           return "(" + buildCond(cx,src(1),j,depth+1) + " ^ "  + buildCond(cx,src(2),j,depth+1) + ")";
  if (m == "S_ANDN2_B64")         return "(" + buildCond(cx,src(1),j,depth+1) + " & ~" + buildCond(cx,src(2),j,depth+1) + ")";
  if (m == "S_MOV_B64" || m == "S_MOV_B32") return buildCond(cx, src(1), j, depth+1);
  return m + "(" + src(1) + "…)";                      // unhandled mask producer — surfaced, not hidden
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <code-object> [func-substr]\n", argv[0]); return 2; }
  std::string want = argc >= 3 ? argv[2] : "";
  SymtabCodeSource *scs = new SymtabCodeSource(argv[1]);
  CodeObject *co = new CodeObject(scs); co->parse();

  for (auto *f : co->funcs()) {
    if (!want.empty() && f->name().find(want) == std::string::npos) continue;
    std::vector<Insn> ins; std::map<Address,int> idxAt;
    std::map<Address,std::string> raw;
    for (auto *b : f->blocks()) { Block::Insns bi; b->getInsns(bi);
      for (auto &kv : bi) raw[kv.first] = kv.second.format(); }
    for (auto &kv : raw) { idxAt[kv.first] = ins.size(); ins.push_back(parseInsn(kv.first, kv.second)); }
    Ctx cx{&ins, &idxAt};

    int nsel = 0, ndiv = 0; bool hdr = false;
    auto header = [&]{ if (!hdr) { printf("\n=== %s ===\n", f->name().c_str()); hdr = true; } };

    for (int i = 0; i < (int)ins.size(); ++i) {
      const std::string &m = ins[i].mnem;
      const auto &o = ins[i].ops;

      // (a) PREDICATION: V_CNDMASK dst, Fval, Tval, MASK  ->  dst = MASK ? Tval : Fval
      if (m.rfind("V_CNDMASK", 0) == 0 && o.size() >= 4) {
        header(); nsel++;
        printf("  [select] 0x%lx  %s = %s ? %s : %s\n", (unsigned long)ins[i].addr,
               o[0].c_str(), buildCond(cx, o[3], i, 0).c_str(), o[2].c_str(), o[1].c_str());
        continue;
      }
      // (b) EXEC DIVERGENCE: S_AND_SAVEEXEC dst, MASK -> enter `if (cond)`;
      //     S_ANDN2_SAVEEXEC -> the complementary `else`. Same evaluator on the mask (SSRC0).
      if (m.rfind("S_AND_SAVEEXEC", 0) == 0 && o.size() >= 2) {
        header(); ndiv++;
        printf("  [if]     0x%lx  if ( %s )\n", (unsigned long)ins[i].addr, buildCond(cx, o[1], i, 0).c_str());
        continue;
      }
      if (m.rfind("S_ANDN2_SAVEEXEC", 0) == 0) {
        header(); printf("  [else]   0x%lx  else / else-if\n", (unsigned long)ins[i].addr);
        continue;
      }
    }
    if (nsel || ndiv) printf("  -> %d select(s) de-predicated, %d divergent branch(es) recovered\n", nsel, ndiv);
  }
  return 0;
}
