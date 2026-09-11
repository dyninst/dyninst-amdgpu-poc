// simtcfg.C — SCRATCH prototype of the typed SIMT-CFG overlay (step 3), on top of the exec-split
// (step 2). Post-parse we (a) split so every EXEC-changer starts a real single-entry block, (b) tag
// each block with a SIMT role from its head instruction, (c) type each edge (IF_TRUE / IF_FALSE /
// THEN_EXIT / ELSE_ENTER / ELSE_SKIP / RECONVERGE), and (d) group condition+flip+reconverge into
// SimtIf instances — the addressable units for instrumentation ("instrument this SimtIf's condition
// block / its THEN edge"). Reuses classify.h for the exec predicates + role classification.
//
//   usage:  ./simtcfg <code-object> [func-substr]           # typed skeleton + SimtIf grouping
//           ./simtcfg <code-object> [func-substr] --dot > g.dot
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <map>
#include "CodeObject.h"
#include "CodeSource.h"
#include "CFG.h"
#include "CFGModifier.h"
#include "Instruction.h"
#include "entryIDs.h"
#include "classify.h"

using namespace Dyninst;
using namespace Dyninst::ParseAPI;
using cfr::MaskOp;

// ---- SIMT role of a block = the role of its head instruction (post-split, that's the whole point) --
enum class Role { Plain, Condition, MaskFlip, Reconverge, Narrow };
static const char* roleName(Role r){
  switch(r){ case Role::Condition:  return "CONDITION";
             case Role::MaskFlip:   return "MASKFLIP";
             case Role::Reconverge: return "RECONVERGE";
             case Role::Narrow:     return "NARROW(&&)";
             default:               return "plain"; }
}
static Role blockRole(Block *b){
  Block::Insns insns; b->getInsns(insns);
  auto it = insns.find(b->start());
  if(it == insns.end()) return Role::Plain;
  switch(cfr::classify(it->second)){
    case MaskOp::Open:    return Role::Condition;
    case MaskOp::Else:
    case MaskOp::Reroute: return Role::MaskFlip;
    case MaskOp::Close:   return Role::Reconverge;
    case MaskOp::Narrow:  return Role::Narrow;
    default:              return Role::Plain;
  }
}

static bool endsInExecz(Block *b){
  Block::Insns insns; b->getInsns(insns);
  if(insns.empty()) return false;
  return insns.rbegin()->second.getOperation().getID() == amdgpu_gfx908_op_S_CBRANCH_EXECZ;
}

// ---- edge typing ---------------------------------------------------------------------------------
enum class ET { IfTrue, IfFalse, ThenExit, ElseEnter, ElseSkip, ElseExit, Reconverge, Fall, Other };
static const char* etName(ET t){
  switch(t){ case ET::IfTrue: return "IF_TRUE";  case ET::IfFalse: return "IF_FALSE";
             case ET::ThenExit: return "THEN_EXIT"; case ET::ElseEnter: return "ELSE_ENTER";
             case ET::ElseSkip: return "ELSE_SKIP"; case ET::ElseExit: return "ELSE_EXIT";
             case ET::Reconverge: return "RECONVERGE";
             case ET::Fall: return "fall"; default: return "other"; }
}
// Type one out-edge of `src` (role rs) to `dst` (role rd). Uses the endpoint roles + whether src
// ends in execz (so the branch-taken edge is the "skip" arm) + the edge's CFG type.
static ET typeEdge(Block *src, Role rs, Edge *e, Role rd){
  const bool execz = endsInExecz(src);
  const EdgeTypeEnum t = e->type();
  if(rs == Role::Condition){
    // execz-taken skips the THEN body (IF_FALSE); the other successor enters it (IF_TRUE).
    if(execz) return (t == COND_TAKEN) ? ET::IfFalse : ET::IfTrue;
    return (rd == Role::Reconverge || rd == Role::MaskFlip) ? ET::IfFalse : ET::IfTrue;
  }
  if(rs == Role::MaskFlip)
    return (rd == Role::Reconverge) ? ET::ElseSkip     // execz over an empty/skipped ELSE
                                    : ET::ElseEnter;   // into the ELSE body
  if(rd == Role::Reconverge) return ET::ThenExit;      // a body arm (then or else) reaching reconverge
  return (t == FALLTHROUGH) ? ET::Fall : ET::Other;
}

// ---- SimtIf: the addressable construct (condition + its arms + flip + reconverge) ----------------
struct SimtIf {
  Block *condition = nullptr, *maskflip = nullptr, *reconverge = nullptr;
  Block *ifTrue = nullptr, *ifFalse = nullptr;   // the CONDITION's two successors (arm entries)
};

static SimtIf buildSimtIf(Function *f, Block *cond){
  SimtIf s; s.condition = cond;
  for(Edge *e : cond->targets()){
    if(!e->trg()) continue;
    ET et = typeEdge(cond, Role::Condition, e, blockRole(e->trg()));
    if(et == ET::IfTrue)  s.ifTrue  = e->trg();
    if(et == ET::IfFalse) s.ifFalse = e->trg();
  }
  // the mask-flip is the else-entry: an IF_FALSE target that is itself a MASKFLIP.
  if(s.ifFalse && blockRole(s.ifFalse) == Role::MaskFlip) s.maskflip = s.ifFalse;
  // Reconvergence: the RECONVERGE-role block at/after the post-dominator (the post-dominator itself
  // can be the maskflip when the else is empty). Walk fallthrough/edges forward to the s_or-exec block.
  Block *pd = f->getImmediatePostDominator(cond);
  std::set<Block*> seen;
  while(pd && !seen.count(pd)){
    seen.insert(pd);
    if(blockRole(pd) == Role::Reconverge){ s.reconverge = pd; break; }
    Block *next = nullptr;                       // follow the unique fallthrough toward reconvergence
    for(Edge *e : pd->targets()) if(e->type() == FALLTHROUGH && e->trg()){ next = e->trg(); break; }
    pd = next;
  }
  return s;
}

int main(int argc, char **argv){
  bool dot = argc >= 2 && std::strcmp(argv[argc-1], "--dot") == 0;
  if(argc < 2){ fprintf(stderr, "usage: %s <code-object> [func-substr] [--dot]\n", argv[0]); return 2; }
  std::string want;
  for(int i = 2; i < argc; ++i){ if(std::strcmp(argv[i], "--dot")) want = argv[i]; }

  SymtabCodeSource *scs = new SymtabCodeSource(argv[1]);
  CodeObject *co = new CodeObject(scs);
  co->parse();

  if(dot) printf("digraph SIMTCFG {\n  node [fontname=\"monospace\",fontsize=10];\n");

  for(Function *f : co->funcs()){
    if(!want.empty() && f->name().find(want) == std::string::npos) continue;

    // The EXEC-split now happens at PARSE time (IA_amdgpu::isModifyExecMask + Parser.C), so every
    // exec-writer is already a block head — the analysis just consumes that CFG (no post-parse split).
    // A block whose head is an exec-writer confirms the parser split is in effect.
    bool anyExec = false;
    for(Block *b : f->blocks()){ Block::Insns in; b->getInsns(in);
      if(!in.empty() && cfr::isExecSplitPoint(in.begin()->second)){ anyExec = true; break; } }
    if(!anyExec) continue;

    std::map<Address,Block*> ord;
    for(Block *b : f->blocks()) ord[b->start()] = b;

    if(dot){
      printf("  subgraph \"cluster_%lx\" { label=\"%s\";\n", (unsigned long)f->addr(), f->name().c_str());
      for(auto &kv : ord){
        Role r = blockRole(kv.second);
        const char *shape = r==Role::Condition?"triangle":r==Role::Reconverge?"invtriangle":
                            r==Role::MaskFlip?"trapezium":r==Role::Narrow?"house":"box";
        printf("    \"%lx\" [shape=%s,label=\"0x%lx\\n%s\"];\n",
               (unsigned long)kv.first, shape, (unsigned long)kv.first, roleName(r));
      }
      printf("  }\n");
      for(auto &kv : ord){ Block *b = kv.second; Role rs = blockRole(b);
        for(Edge *e : b->targets()){ if(!e->trg()) continue;
          ET et = typeEdge(b, rs, e, blockRole(e->trg()));
          const char *col = et==ET::IfTrue?"darkgreen":et==ET::IfFalse?"red":
                            et==ET::ElseEnter?"blue":et==ET::ThenExit?"gray40":"black";
          printf("  \"%lx\" -> \"%lx\" [color=%s,label=\"%s\"];\n",
                 (unsigned long)b->start(), (unsigned long)e->trg()->start(), col, etName(et)); } }
      continue;
    }

    printf("\n========== %s ==========\n", f->name().c_str());
    printf("  --- typed blocks + edges ---\n");
    for(auto &kv : ord){
      Block *b = kv.second; Role r = blockRole(b);
      printf("    0x%-6lx  %-11s ->", (unsigned long)b->start(), roleName(r));
      for(Edge *e : b->targets()) if(e->trg())
        printf("  0x%lx[%s]", (unsigned long)e->trg()->start(), etName(typeEdge(b, r, e, blockRole(e->trg()))));
      printf("\n");
    }
    printf("  --- SimtIf constructs ---\n");
    int k = 0;
    for(auto &kv : ord){
      if(blockRole(kv.second) != Role::Condition) continue;
      SimtIf s = buildSimtIf(f, kv.second);
      char flip[24];
      if(s.maskflip) snprintf(flip, sizeof flip, "0x%lx", (unsigned long)s.maskflip->start());
      else           snprintf(flip, sizeof flip, "none");
      char reconv[24];
      if(s.reconverge) snprintf(reconv, sizeof reconv, "0x%lx", (unsigned long)s.reconverge->start());
      else             snprintf(reconv, sizeof reconv, "?");
      printf("    SimtIf #%d  condition@0x%lx  IF_TRUE->0x%lx  IF_FALSE->0x%lx  maskflip@%s  reconverge@%s\n",
             k++, (unsigned long)s.condition->start(),
             s.ifTrue ? (unsigned long)s.ifTrue->start() : 0,
             s.ifFalse ? (unsigned long)s.ifFalse->start() : 0, flip, reconv);
    }
  }
  if(dot) printf("}\n");
  return 0;
}
