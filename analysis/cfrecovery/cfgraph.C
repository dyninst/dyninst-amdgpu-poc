// cfgraph.C — a CFG-to-Graphviz dumper (in the spirit of Dyninst's CFGraph example) extended with our
// SIMT analysis: each basic block is shaped by its EXEC-mask role from classify(). Divergence points
// (s_and_saveexec — narrow EXEC, open a divergent region) are drawn as UPWARD triangles; reconvergence
// points (s_or_b64 exec — restore EXEC) as DOWNWARD triangles; else-arms and `&&`-narrows get their own
// shapes. Each divergence header is also paired to its post-dominator reconvergence with a dashed edge.
//
//   usage:  ./cfgraph <code-object> [func-substr]  > out.dot   &&   dot -Tpng out.dot -o out.png
#include <cstdio>
#include <string>
#include <set>
#include "CodeObject.h"
#include "CodeSource.h"
#include "CFG.h"
#include "Instruction.h"
#include "classify.h"

using namespace Dyninst;
using namespace Dyninst::ParseAPI;
using cfr::MaskOp;

// The divergence role of a block = the strongest EXEC-mask op it contains.
enum class Role { Plain, Diverge, Converge, Else, Narrow };

static Role blockRole(Block *b){
  Block::Insns insns; b->getInsns(insns);
  bool open = false, close = false, els = false, narrow = false;
  for(const auto &kv : insns){
    switch(cfr::classify(kv.second)){
      case MaskOp::Open:    open   = true; break;
      case MaskOp::Close:   close  = true; break;
      case MaskOp::Else:
      case MaskOp::Reroute: els    = true; break;
      case MaskOp::Narrow:  narrow = true; break;
      default: break;
    }
  }
  if(open)   return Role::Diverge;    // s_and_saveexec — opens a divergent region
  if(close)  return Role::Converge;   // s_or_b64 exec  — reconvergence
  if(els)    return Role::Else;       // complement flip — the else arm
  if(narrow) return Role::Narrow;     // s_and exec     — && narrowing
  return Role::Plain;
}

static const char *nodeStyle(Role r){
  switch(r){
    case Role::Diverge:  return "shape=triangle,    style=filled, fillcolor=\"#ffd28a\"";  // up   = diverge
    case Role::Converge: return "shape=invtriangle, style=filled, fillcolor=\"#a9d6ff\"";  // down = reconverge
    case Role::Else:     return "shape=trapezium,   style=filled, fillcolor=\"#e8e0ff\"";
    case Role::Narrow:   return "shape=house,       style=filled, fillcolor=\"#fff0b3\"";
    default:             return "shape=box";
  }
}
static const char *roleTag(Role r){
  switch(r){
    case Role::Diverge:  return "diverge";
    case Role::Converge: return "reconverge";
    case Role::Else:     return "else";
    case Role::Narrow:   return "&&";
    default:             return "";
  }
}
static const char *edgeStyle(EdgeTypeEnum t){
  switch(t){
    case COND_TAKEN:     return "color=darkgreen, label=\"T\"";
    case COND_NOT_TAKEN: return "color=red, label=\"F\"";
    case FALLTHROUGH:    return "color=gray40";
    case CALL:           return "color=blue, style=dashed, label=\"call\"";
    case RET:            return "color=blue, style=dotted";
    case INDIRECT:       return "color=purple, style=dashed";
    default:             return "color=black";
  }
}

int main(int argc, char **argv){
  if(argc < 2){ fprintf(stderr, "usage: %s <code-object> [func-substr]  > out.dot\n", argv[0]); return 2; }
  const std::string want = argc >= 3 ? argv[2] : "";

  SymtabCodeSource *scs = new SymtabCodeSource(argv[1]);
  CodeObject *co = new CodeObject(scs);
  co->parse();

  printf("digraph CFG {\n");
  printf("  graph [rankdir=TB];\n");
  printf("  node  [fontname=\"monospace\", fontsize=10];\n");
  printf("  edge  [fontname=\"monospace\", fontsize=9];\n");
  // legend
  printf("  subgraph cluster_legend {\n    label=\"legend\"; color=gray80; fontsize=9;\n");
  printf("    L0 [%s, label=\"diverge\\n(saveexec)\"];\n", nodeStyle(Role::Diverge));
  printf("    L1 [%s, label=\"reconverge\\n(or exec)\"];\n", nodeStyle(Role::Converge));
  printf("    L2 [%s, label=\"else\"];\n", nodeStyle(Role::Else));
  printf("    L3 [%s, label=\"&&\"];\n", nodeStyle(Role::Narrow));
  printf("    L0 -> L1 [style=invis]; L1 -> L2 [style=invis]; L2 -> L3 [style=invis];\n  }\n");

  for(Function *f : co->funcs()){
    if(!want.empty() && f->name().find(want) == std::string::npos) continue;

    // when no function was named, emit only the ones that actually diverge (keeps the graph readable)
    bool anyDiv = false;
    for(Block *b : f->blocks()) if(blockRole(b) == Role::Diverge){ anyDiv = true; break; }
    if(want.empty() && !anyDiv) continue;

    printf("  subgraph \"cluster_%lx\" {\n    label=\"%s\";\n    color=gray70;\n",
           (unsigned long)f->addr(), f->name().c_str());

    std::set<Address> seen;
    for(Block *b : f->blocks()){
      if(!seen.insert(b->start()).second) continue;               // dedup shared blocks
      const Role role = blockRole(b);
      const char *tag = roleTag(role);
      if(*tag)
        printf("    \"%lx\" [%s, label=\"0x%lx\\n%s\"];\n",
               (unsigned long)b->start(), nodeStyle(role), (unsigned long)b->start(), tag);
      else
        printf("    \"%lx\" [%s, label=\"0x%lx\"];\n",
               (unsigned long)b->start(), nodeStyle(role), (unsigned long)b->start());
    }
    printf("  }\n");

    // edges (emitted outside the cluster), colored by CFG edge type
    seen.clear();
    for(Block *b : f->blocks()){
      if(!seen.insert(b->start()).second) continue;
      for(Edge *e : b->targets())
        if(e->trg())
          printf("  \"%lx\" -> \"%lx\" [%s];\n",
                 (unsigned long)b->start(), (unsigned long)e->trg()->start(), edgeStyle(e->type()));

      // pair each divergence header with its structural reconvergence (immediate post-dominator)
      if(blockRole(b) == Role::Diverge){
        Block *pd = f->getImmediatePostDominator(b);
        if(pd)
          printf("  \"%lx\" -> \"%lx\" [style=dashed, color=orange, constraint=false, label=\"reconv\"];\n",
                 (unsigned long)b->start(), (unsigned long)pd->start());
      }
    }
  }
  printf("}\n");
  return 0;
}
