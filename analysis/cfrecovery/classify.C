// classify.C — SEMANTIC classification of EXEC-mask / predication ops, the foundation
// of the CFRecovery API. No full-instruction string parsing: classification is by
// entryID (Operation::getID()), operand ROLES (Operand::isRead/isWritten), and register
// IDENTITY (MachRegister ==). Replaces the throwaway `format()` string matching.
#include <cstdio>
#include <string>
#include <set>
#include <map>
#include "CodeObject.h"
#include "CodeSource.h"
#include "Instruction.h"
#include "Operand.h"
#include "Register.h"
#include "dyn_regs.h"
#include "entryIDs.h"
#include "registers/AMDGPU/amdgpu_gfx908_regs.h"

using namespace Dyninst;
using namespace Dyninst::ParseAPI;
using namespace Dyninst::InstructionAPI;
namespace R = Dyninst::amdgpu_gfx908;

enum class MaskOp { None, Open, Close, Else, Select, Call, Compare };
static const char *name(MaskOp m){
  switch(m){case MaskOp::Open:return"OPEN ";case MaskOp::Close:return"CLOSE";case MaskOp::Else:return"ELSE ";
  case MaskOp::Select:return"SELECT";case MaskOp::Call:return"CALL ";case MaskOp::Compare:return"CMP  ";default:return"·    ";}
}

// --- semantic register queries (identity, not text) --------------------------------
static bool writesReg(const Instruction &in, MachRegister r){
  std::set<RegisterAST::Ptr> w; in.getWriteSet(w);
  for (auto &x : w) if (x->getID() == r) return true;
  return false;
}
static bool writesExec(const Instruction &in){ return writesReg(in,R::exec_lo) || writesReg(in,R::exec_hi); }

// --- classify() --------------------------------------------------------------------
static MaskOp classify(const Instruction &in){
  switch (in.getOperation().getID()){
    // OPEN: narrows exec, saves old exec (SDST written), reads a condition mask (SSRC0).
    case amdgpu_gfx908_op_S_AND_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_OR_SAVEEXEC_B64:
      return MaskOp::Open;
    // ELSE / complement flip.
    case amdgpu_gfx908_op_S_ANDN2_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_XOR_SAVEEXEC_B64:
      return MaskOp::Else;
    // Context-dependent: only a CLOSE/ELSE when it actually writes EXEC (else plain SGPR alu).
    case amdgpu_gfx908_op_S_OR_B64:
    case amdgpu_gfx908_op_S_MOV_B64:
      return writesExec(in) ? MaskOp::Close : MaskOp::None;
    case amdgpu_gfx908_op_S_XOR_B64:
    case amdgpu_gfx908_op_S_ANDN2_B64:
      return writesExec(in) ? MaskOp::Else : MaskOp::None;
    case amdgpu_gfx908_op_V_CNDMASK_B32:
      return MaskOp::Select;
    case amdgpu_gfx908_op_S_SWAPPC_B64:
      return MaskOp::Call;
    default:
      // Compare: opcode-only mnemonic (no operands) — used only to tag the predicate leaf.
      if (in.getOperation().format().rfind("V_CMP", 0) == 0) return MaskOp::Compare;
      return MaskOp::None;
  }
}

// For an OPEN: semantically extract (condition mask = the read operand that isn't exec)
// and (save reg = the written SGPR that isn't exec) via operand roles, not position.
static std::string readMaskOperand(const Instruction &in){
  std::vector<Operand> ops; in.getOperands(ops);
  for (auto &o : ops){
    if (!o.isRead() || o.isWritten()) continue;        // a pure source operand (SSRC0)
    std::string s = o.getValue()->format();            // per-operand (keeps VCC intact, not flattened)
    if (s=="EXEC"||s=="EXEC_LO"||s=="EXEC_HI"||s=="SCC") continue;
    return s;                                           // the condition mask
  }
  return "?";
}

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: %s <co> [func-substr]\n",argv[0]); return 2; }
  std::string want=argc>=3?argv[2]:"";
  SymtabCodeSource *scs=new SymtabCodeSource(argv[1]); CodeObject *co=new CodeObject(scs); co->parse();
  for(auto *f:co->funcs()){
    if(!want.empty() && f->name().find(want)==std::string::npos) continue;
    std::map<Dyninst::Address,Instruction> all;
    for(auto *b:f->blocks()){ Block::Insns bi; b->getInsns(bi); for(auto &kv:bi) all.insert(kv); }
    bool hdr=false;
    for(std::map<Dyninst::Address,Instruction>::iterator it=all.begin(); it!=all.end(); ++it){
      MaskOp m=classify(it->second);
      if(m==MaskOp::None) continue;
      if(!hdr){ printf("\n=== %s ===\n",f->name().c_str()); hdr=true; }
      printf("  0x%-6lx %s  %s", (unsigned long)it->first, name(m), it->second.format().c_str());
      if(m==MaskOp::Open) printf("        [cond mask = %s]", readMaskOperand(it->second).c_str());
      printf("\n");
    }
  }
  return 0;
}
