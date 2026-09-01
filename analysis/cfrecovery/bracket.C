// bracket.C — forward mask-stack bracketing pass. Walks a function's EXEC-mask ops as a
// stack keyed on the save register, turning flat divergence idioms into a NESTED
// if/elseif/else tree with recovered conditions (reusing the mask evaluator). Bodies
// (s_swappc calls) attach at their nesting depth.
//
//   S_AND_SAVEEXEC s[X], MASK        -> OPEN  `if (cond)`   (push region s[X])
//   S_ANDN2_SAVEEXEC s[X], s[X]      -> ELSE  of region s[X] (same reg = complement flip)
//   S_ANDN2_SAVEEXEC s[X], s[Y]      -> chained ELSE-IF (routes remaining lanes; flagged)
//   S_XOR_B64 exec, exec, s[X]       -> ELSE  of region s[X]
//   S_OR_B64 exec, exec, s[X] | S_MOV_B64 exec, s[X] -> CLOSE region s[X] (pop)
//
// Scope (MVP): reaching-def for conditions = nearest prior writer by address (exact for
// straight-line regions). Save-register match is best-effort (pop nearest matching frame).
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "CodeObject.h"
#include "CodeSource.h"
#include "Instruction.h"
using namespace Dyninst; using namespace Dyninst::ParseAPI; using namespace Dyninst::InstructionAPI;

struct Insn { Address addr; std::string mnem; std::vector<std::string> ops; };

static std::vector<std::string> splitOps(const std::string &rest) {
  std::vector<std::string> v; std::string cur; int depth = 0;
  for (char c : rest) {
    if (c=='[') depth++; else if (c==']') depth--;
    if (c==',' && depth==0) { while(!cur.empty()&&cur.front()==' ')cur.erase(cur.begin());
      while(!cur.empty()&&cur.back()==' ')cur.pop_back(); if(!cur.empty())v.push_back(cur); cur.clear(); }
    else cur+=c;
  }
  while(!cur.empty()&&cur.front()==' ')cur.erase(cur.begin());
  while(!cur.empty()&&cur.back()==' ')cur.pop_back(); if(!cur.empty())v.push_back(cur);
  return v;
}
static Insn parseInsn(Address a, const std::string &fmt) {
  Insn in; in.addr=a; size_t sp=fmt.find(' ');
  if (sp==std::string::npos){ in.mnem=fmt; return in; }
  in.mnem=fmt.substr(0,sp); in.ops=splitOps(fmt.substr(sp+1)); return in;
}
static std::string cmpOp(const std::string &m){
  std::vector<std::string> p; std::stringstream ss(m); std::string t; while(std::getline(ss,t,'_'))p.push_back(t);
  std::string c=p.size()>2?p[2]:"?";
  if(c=="LT")return"<"; if(c=="GT")return">"; if(c=="LE"||c=="NGT")return"<="; if(c=="GE"||c=="NLT")return">=";
  if(c=="EQ")return"=="; if(c=="NE"||c=="LG"||c=="NEQ")return"!="; return "."+c+".";
}
struct Ctx { std::vector<Insn> *ins; };
static int defOf(Ctx &cx,const std::string &reg,int before){
  for(int j=before-1;j>=0;--j) if(!(*cx.ins)[j].ops.empty() && (*cx.ins)[j].ops[0]==reg) return j; return -1;
}
static std::string buildCond(Ctx &cx,const std::string &reg,int before,int depth){
  if(depth>32) return reg+"…";
  int j=defOf(cx,reg,before); if(j<0) return reg;
  const Insn &d=(*cx.ins)[j]; const std::string &m=d.mnem;
  auto src=[&](int i){ return i<(int)d.ops.size()?d.ops[i]:std::string("?"); };
  if(m.rfind("V_CMP",0)==0) return "("+src(1)+" "+cmpOp(m)+" "+src(2)+")";
  if(m=="S_AND_B64")   return "("+buildCond(cx,src(1),j,depth+1)+" & " +buildCond(cx,src(2),j,depth+1)+")";
  if(m=="S_OR_B64")    return "("+buildCond(cx,src(1),j,depth+1)+" | " +buildCond(cx,src(2),j,depth+1)+")";
  if(m=="S_XOR_B64")   return "("+buildCond(cx,src(1),j,depth+1)+" ^ " +buildCond(cx,src(2),j,depth+1)+")";
  if(m=="S_MOV_B64"||m=="S_MOV_B32") return buildCond(cx,src(1),j,depth+1);
  return m;
}

struct Frame { std::string save; int depth; };

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: %s <co> [func-substr]\n",argv[0]); return 2; }
  std::string want=argc>=3?argv[2]:"";
  SymtabCodeSource *scs=new SymtabCodeSource(argv[1]); CodeObject *co=new CodeObject(scs); co->parse();
  for(auto *f:co->funcs()){
    if(!want.empty() && f->name().find(want)==std::string::npos) continue;
    std::map<Address,std::string> raw;
    for(auto *b:f->blocks()){ Block::Insns bi; b->getInsns(bi); for(auto &kv:bi) raw[kv.first]=kv.second.format(); }
    std::vector<Insn> ins; for(auto &kv:raw) ins.push_back(parseInsn(kv.first,kv.second));
    Ctx cx{&ins};
    // any EXEC brackets here?
    bool any=false; for(auto&I:ins) if(I.mnem.find("SAVEEXEC")!=std::string::npos){any=true;break;}
    if(!any) continue;

    printf("\n=== %s ===\n", f->name().c_str());
    std::vector<Frame> stk; int depth=0;
    auto ind=[&](int d){ for(int i=0;i<d;i++) printf("  "); };
    for(int i=0;i<(int)ins.size();++i){
      const std::string &m=ins[i].mnem; const auto &o=ins[i].ops; Address a=ins[i].addr;
      auto isExecRestore=[&](const std::string&mm,const std::vector<std::string>&oo,std::string&sv){
        if(mm=="S_OR_B64"&&oo.size()>=3&&oo[0]=="EXEC"&&oo[1]=="EXEC"){sv=oo[2];return true;}
        if(mm=="S_MOV_B64"&&oo.size()>=2&&oo[0]=="EXEC"&&oo[1].rfind("S[",0)==0){sv=oo[1];return true;}
        return false; };
      std::string sv;
      if(m.rfind("S_AND_SAVEEXEC",0)==0 && o.size()>=2){                 // OPEN
        std::string cond=buildCond(cx,o[1],i,0);
        ind(depth); printf("if ( %s ) {            // @0x%lx  save %s\n",cond.c_str(),(unsigned long)a,o.size()>0?o[0].c_str():"?");
        stk.push_back({o.size()>0?o[0]:std::string(),depth}); depth++;
      } else if(m.rfind("S_ANDN2_SAVEEXEC",0)==0){                        // ELSE / chained else-if
        bool sameReg = o.size()>=2 && o[0]==o[1];
        if(depth>0) depth--;
        ind(depth); printf("} else%s {            // @0x%lx\n", sameReg?"":" /*else-if chain*/",(unsigned long)a);
        depth++;
      } else if(m=="S_XOR_B64" && o.size()>=3 && o[0]=="EXEC" && o[1]=="EXEC"){ // ELSE
        if(depth>0) depth--; ind(depth); printf("} else {              // @0x%lx (xor)\n",(unsigned long)a); depth++;
      } else if(isExecRestore(m,o,sv)){                                  // CLOSE
        if(depth>0) depth--;
        // best-effort pop of the matching frame
        if(!stk.empty()){ if(stk.back().save!=sv){ for(int k=(int)stk.size()-1;k>=0;--k) if(stk[k].save==sv){ stk.erase(stk.begin()+k); break; } } else stk.pop_back(); }
        ind(depth); printf("}                     // @0x%lx  close %s\n",(unsigned long)a,sv.c_str());
      } else if(m=="S_SWAPPC_B64"){                                      // body: a call
        ind(depth); printf("call();               // @0x%lx\n",(unsigned long)a);
      }
    }
    while(depth-->0){ ind(depth); printf("}                     // (implicit close at fn end)\n"); }
  }
  return 0;
}
