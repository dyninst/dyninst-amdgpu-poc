// classify.h — semantic classification of AMDGPU EXEC-mask / predication ops, keyed on entryID +
// register identity (no format-string parsing). This is the shared foundation of the CF-recovery
// analysis; structure.C and classify.C currently carry their own in-file copies — consolidate them
// onto this header when convenient. gfx908-only for now (arch dispatch is an integration-time item).
#ifndef CFR_CLASSIFY_H
#define CFR_CLASSIFY_H

#include <string>
#include <set>
#include <vector>
#include "Instruction.h"
#include "Operand.h"
#include "Register.h"
#include "entryIDs.h"
#include "registers/AMDGPU/amdgpu_gfx908_regs.h"

namespace cfr {
using namespace Dyninst;
using namespace Dyninst::InstructionAPI;
namespace R = Dyninst::amdgpu_gfx908;

enum class MaskOp { None, Open, Close, Else, Reroute, Narrow, Select, Call, Compare };

// True if the instruction writes machine register `r` (identity match on the write set).
inline bool writesReg(const Instruction &in, MachRegister r){
  std::set<RegisterAST::Ptr> written; in.getWriteSet(written);
  for(const auto &w : written) if(w->getID() == r) return true;
  return false;
}
// True if the instruction writes EXEC (either half of the 64-bit mask). This is the general
// "changes the active-lane set" test — includes saveexec, or/xor/and exec, mov exec, etc.
inline bool isGeneralWriteExec(const Instruction &in){ return writesReg(in, R::exec_lo) || writesReg(in, R::exec_hi); }
inline bool writesExec(const Instruction &in){ return isGeneralWriteExec(in); }   // legacy name

// The saveexec family — the atomic mask-narrow/flip ops, recognizable by entryID ALONE (no operand
// inspection needed, unlike isGeneralWriteExec). A subset of isGeneralWriteExec.
inline bool isModifyExecMask(const Instruction &in){
  switch(in.getOperation().getID()){
    case amdgpu_gfx908_op_S_AND_SAVEEXEC_B64:  case amdgpu_gfx908_op_S_OR_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_XOR_SAVEEXEC_B64:  case amdgpu_gfx908_op_S_ANDN2_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_ORN2_SAVEEXEC_B64: case amdgpu_gfx908_op_S_NAND_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_NOR_SAVEEXEC_B64:  case amdgpu_gfx908_op_S_XNOR_SAVEEXEC_B64:
      return true;
    default: return false;
  }
}

// SIMT block-split trigger: an EXEC-changer that should START its own block, so the SIMT-CONDITION
// (saveexec head), MASKFLIP, and RECONVERGE become real single-entry blocks. Every exec-writer
// qualifies EXCEPT a plain mov-into-EXEC (the owner's "except mov": a benign save/restore, not a
// divergence transition). NOTE (design, owner): consecutive exec-writers stay SEPARATE blocks — each
// is a distinct mask transition / instrumentation point. (A future gfx90a-only coalescing of a
// single flip pair would require the intermediate EXEC to be dead AND the 2nd writer to have only a
// fallthrough predecessor; not applicable on gfx908.)
inline bool isExecSplitPoint(const Instruction &in){
  const entryID id = in.getOperation().getID();
  if(id == amdgpu_gfx908_op_S_MOV_B64 && isGeneralWriteExec(in)) return false;   // except mov→exec
  return isModifyExecMask(in) || isGeneralWriteExec(in);
}

// A saveexec whose dest (SDST) equals its source (SSRC0) — a clean complement flip (plain `else`),
// vs different registers, which route the REMAINING lanes to the next case.
inline bool isSelfFlip(const Instruction &in){
  std::vector<Operand> ops; in.getOperands(ops);
  return ops.size() >= 2 && ops[0].getValue()->format() == ops[1].getValue()->format();
}

// Classify one instruction's role in the EXEC-mask / predication algebra.
inline MaskOp classify(const Instruction &in){
  switch(in.getOperation().getID()){
    case amdgpu_gfx908_op_S_AND_SAVEEXEC_B64:                       // narrows EXEC, saves old — opens an `if`
      return MaskOp::Open;
    case amdgpu_gfx908_op_S_OR_SAVEEXEC_B64:                        // SI_ELSE entry + the boolean-combine
    case amdgpu_gfx908_op_S_XOR_SAVEEXEC_B64:                       // saveexec fuser forms
    case amdgpu_gfx908_op_S_ORN2_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_NAND_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_NOR_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_XNOR_SAVEEXEC_B64:
      return MaskOp::Else;
    case amdgpu_gfx908_op_S_ANDN2_SAVEEXEC_B64:                     // same-reg → else; diff-reg → reroute
      return isSelfFlip(in) ? MaskOp::Else : MaskOp::Reroute;
    case amdgpu_gfx908_op_S_OR_B64:                                 // context-dependent: CLOSE only if it
    case amdgpu_gfx908_op_S_MOV_B64:                                // actually writes EXEC (else plain SGPR alu)
      return writesExec(in) ? MaskOp::Close : MaskOp::None;
    case amdgpu_gfx908_op_S_XOR_B64:
    case amdgpu_gfx908_op_S_ANDN2_B64:
      return writesExec(in) ? MaskOp::Else : MaskOp::None;
    case amdgpu_gfx908_op_S_AND_B64:                                // `&&` short-circuit narrowing of EXEC
      return writesExec(in) ? MaskOp::Narrow : MaskOp::None;
    case amdgpu_gfx908_op_V_CNDMASK_B32:                            // per-lane data select (if-conversion)
      return MaskOp::Select;
    case amdgpu_gfx908_op_S_SWAPPC_B64:                             // indirect call
      return MaskOp::Call;
    default: {
      const std::string op = in.getOperation().format();           // ~200 compare variants: mnemonic prefix
      if(op.rfind("V_CMP", 0) == 0 || op.rfind("S_CMP", 0) == 0) return MaskOp::Compare;
      return MaskOp::None;
    }
  }
}

} // namespace cfr
#endif // CFR_CLASSIFY_H
