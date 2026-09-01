// structure.C — StructureAnalysis: the CFRecovery region tree, built on semantic
// classify() (no string matching for structure), with a post-dominance cross-check on
// each reconvergence. Produces a queryable Region tree (regionOf(block), condition()).
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <algorithm>
#include "CodeObject.h"
#include "CodeSource.h"
#include "Instruction.h"
#include "Operand.h"
#include "Register.h"
#include "dyn_regs.h"
#include "entryIDs.h"
#include "registers/AMDGPU/amdgpu_gfx908_regs.h"
#include "Symtab.h"
#include "Symbol.h"
#include "Region.h"
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cctype>
using namespace Dyninst; using namespace Dyninst::ParseAPI; using namespace Dyninst::InstructionAPI;
namespace R = Dyninst::amdgpu_gfx908;   // gfx908-only for now; arch-generic is an integration-time item

// ---------- minimal MessagePack reader (for the AMDGPU metadata note, mechanism B) --------------
// The kernel metadata note (NT_AMDGPU_METADATA) is msgpack-encoded; we decode just enough to read
// each kernel's hidden-argument table (.value_kind + .offset). Only the subset of msgpack the AMDGPU
// streamer emits is handled (fixint/uint/int, fix/8/16/32 str, fix/16/32 array + map, nil/bool).
struct MsgVal {
  enum T { NIL, BOOL, INT, STR, ARR, MAP } t = NIL;
  long long i = 0; std::string s;
  std::vector<MsgVal> arr;
  std::vector<std::pair<MsgVal,MsgVal>> map;
};
static const unsigned char* mpParse(const unsigned char* p, const unsigned char* end, MsgVal& out){
  if(p >= end) return nullptr;
  auto be = [&](int n)->unsigned long long { unsigned long long v = 0; for(int k=0;k<n;k++) v=(v<<8)|p[k]; return v; };
  unsigned char c = *p++;
  if(c <= 0x7f){ out.t=MsgVal::INT; out.i=c; return p; }                       // positive fixint
  if(c >= 0xe0){ out.t=MsgVal::INT; out.i=(signed char)c; return p; }          // negative fixint
  if((c & 0xe0) == 0xa0){ int n=c&0x1f; if(p+n>end) return nullptr;            // fixstr
    out.t=MsgVal::STR; out.s.assign((const char*)p,n); return p+n; }
  if((c & 0xf0) == 0x90){ int n=c&0x0f; out.t=MsgVal::ARR;                     // fixarray
    for(int k=0;k<n;k++){ MsgVal e; p=mpParse(p,end,e); if(!p) return nullptr; out.arr.push_back(std::move(e)); } return p; }
  if((c & 0xf0) == 0x80){ int n=c&0x0f; out.t=MsgVal::MAP;                     // fixmap
    for(int k=0;k<n;k++){ MsgVal K,V; p=mpParse(p,end,K); if(!p) return nullptr; p=mpParse(p,end,V); if(!p) return nullptr;
      out.map.emplace_back(std::move(K),std::move(V)); } return p; }
  switch(c){
    case 0xc0: out.t=MsgVal::NIL; return p;
    case 0xc2: out.t=MsgVal::BOOL; out.i=0; return p;
    case 0xc3: out.t=MsgVal::BOOL; out.i=1; return p;
    case 0xcc: if(p+1>end)return nullptr; out.t=MsgVal::INT; out.i=be(1); return p+1;
    case 0xcd: if(p+2>end)return nullptr; out.t=MsgVal::INT; out.i=be(2); return p+2;
    case 0xce: if(p+4>end)return nullptr; out.t=MsgVal::INT; out.i=be(4); return p+4;
    case 0xcf: if(p+8>end)return nullptr; out.t=MsgVal::INT; out.i=(long long)be(8); return p+8;
    case 0xd0: if(p+1>end)return nullptr; out.t=MsgVal::INT; out.i=(signed char)p[0]; return p+1;
    case 0xd1: if(p+2>end)return nullptr; out.t=MsgVal::INT; out.i=(short)be(2); return p+2;
    case 0xd2: if(p+4>end)return nullptr; out.t=MsgVal::INT; out.i=(int)be(4); return p+4;
    case 0xd3: if(p+8>end)return nullptr; out.t=MsgVal::INT; out.i=(long long)be(8); return p+8;
    case 0xd9: { if(p+1>end)return nullptr; int n=p[0]; p+=1; if(p+n>end)return nullptr;   // str8
                 out.t=MsgVal::STR; out.s.assign((const char*)p,n); return p+n; }
    case 0xda: { if(p+2>end)return nullptr; int n=be(2); p+=2; if(p+n>end)return nullptr;  // str16
                 out.t=MsgVal::STR; out.s.assign((const char*)p,n); return p+n; }
    case 0xdb: { if(p+4>end)return nullptr; long n=be(4); p+=4; if(p+n>end)return nullptr; // str32
                 out.t=MsgVal::STR; out.s.assign((const char*)p,n); return p+n; }
    case 0xdc: { if(p+2>end)return nullptr; int n=be(2); p+=2; out.t=MsgVal::ARR;          // array16
                 for(int k=0;k<n;k++){ MsgVal e; p=mpParse(p,end,e); if(!p)return nullptr; out.arr.push_back(std::move(e)); } return p; }
    case 0xdd: { if(p+4>end)return nullptr; long n=be(4); p+=4; out.t=MsgVal::ARR;         // array32
                 for(long k=0;k<n;k++){ MsgVal e; p=mpParse(p,end,e); if(!p)return nullptr; out.arr.push_back(std::move(e)); } return p; }
    case 0xde: { if(p+2>end)return nullptr; int n=be(2); p+=2; out.t=MsgVal::MAP;          // map16
                 for(int k=0;k<n;k++){ MsgVal K,V; p=mpParse(p,end,K); if(!p)return nullptr; p=mpParse(p,end,V); if(!p)return nullptr;
                   out.map.emplace_back(std::move(K),std::move(V)); } return p; }
    case 0xdf: { if(p+4>end)return nullptr; long n=be(4); p+=4; out.t=MsgVal::MAP;         // map32
                 for(long k=0;k<n;k++){ MsgVal K,V; p=mpParse(p,end,K); if(!p)return nullptr; p=mpParse(p,end,V); if(!p)return nullptr;
                   out.map.emplace_back(std::move(K),std::move(V)); } return p; }
    default: return nullptr;   // float/bin/ext not emitted here
  }
}
// Value lookup in a msgpack map by string key.
static const MsgVal* mpGet(const MsgVal& m, const char* key){
  if(m.t != MsgVal::MAP) return nullptr;
  for(const auto& kv : m.map) if(kv.first.t==MsgVal::STR && kv.first.s==key) return &kv.second;
  return nullptr;
}
// AMDGPU hidden-arg .value_kind → friendly implicit-arg name ("" if not a hidden arg).
static std::string prettyHidden(const std::string& vk){
  auto dim = [&](const std::string& base)->std::string {
    if(vk.size()>=2 && vk[vk.size()-2]=='_'){ char c=vk.back(); if(c=='x'||c=='y'||c=='z') return base+"."+c; }
    return base; };
  if(vk.rfind("hidden_group_size",0)==0)    return dim("blockDim");   // workgroup size = blockDim
  if(vk.rfind("hidden_block_count",0)==0)   return dim("gridDim");    // #workgroups = gridDim
  if(vk.rfind("hidden_remainder",0)==0)     return dim("remainder");
  if(vk.rfind("hidden_global_offset",0)==0) return dim("globalOffset");
  if(vk.rfind("hidden_",0)==0)              return vk.substr(7);      // grid_dims, heap_v1, …
  return "";
}
// ---- register-identity helpers (decompose/classify via MachRegister, not Operand::format() strings) ----
// Canonical key for one register = its uppercased name — matches Operand::format() (s4→"S4", v0→"V0"),
// so it stays consistent with the ABI seeds without any string parsing.
static std::string regKey(MachRegister r){ std::string n = r.name(); for(char &c : n) c = std::toupper((unsigned char)c); return n; }
static bool isScalarReg(MachRegister r){ return r.regClass() == amdgpu_gfx908::s0.regClass(); }
// A "data" register we track: SGPR/VGPR only, excluding the mask/special regs (exec/vcc/scc/m0/waitcnt).
static bool isDataReg(MachRegister r){
  const auto sc = amdgpu_gfx908::s0.regClass(), vc = amdgpu_gfx908::v0.regClass();
  if(r.regClass() != sc && r.regClass() != vc) return false;
  if(r == R::exec_lo || r == R::exec_hi || r == R::vcc_lo || r == R::vcc_hi) return false;
  return r.name() != "scc";
}
// The data registers an operand touches, sorted by number — front() is the low 32-bit half of a pair.
static std::vector<MachRegister> dataRegs(const std::set<RegisterAST::Ptr> &rs){
  std::vector<MachRegister> v;
  for(const auto &x : rs){ MachRegister m = x->getID(); if(isDataReg(m)) v.push_back(m); }
  std::sort(v.begin(), v.end(), [](MachRegister a, MachRegister b){ return a.val() < b.val(); });
  return v;
}
// hsa_kernel_dispatch_packet_t field at byte `off` (mechanism A — fixed HSA ABI, KD-alone). A dword
// load at 0x04 packs blockDim.x|blockDim.y; we label it by the low half (the common index-math use).
static std::string packetField(long long off){
  switch(off){
    case 0x04: return "blockDim.x"; case 0x06: return "blockDim.y"; case 0x08: return "blockDim.z";
    case 0x0c: return "gridSize.x"; case 0x10: return "gridSize.y"; case 0x14: return "gridSize.z";
    default:   return "";           // gridSize = #workitems (= gridDim*blockDim); header/private/etc. unlabeled
  }
}

// ---------- semantic classify() (from classify.C) ----------------------------------
enum class MaskOp { None, Open, Close, Else, Reroute, Narrow, Select, Call, Compare };
// True if the instruction writes machine register `r` (identity match on the write set).
static bool writesReg(const Instruction &in, MachRegister r){
  std::set<RegisterAST::Ptr> written;
  in.getWriteSet(written);
  for(const auto &w : written)
    if(w->getID() == r) return true;
  return false;
}

// True if the instruction writes EXEC (either half of the 64-bit mask).
static bool writesExec(const Instruction &in){
  return writesReg(in, R::exec_lo) || writesReg(in, R::exec_hi);
}

// True if a saveexec's dest (SDST) and source (SSRC0) are the same register — a clean complement
// flip (a plain `else`) — vs different registers, which route the REMAINING lanes to the next case.
static bool isSelfFlip(const Instruction &in){
  std::vector<Operand> ops; in.getOperands(ops);
  return ops.size() >= 2 && ops[0].getValue()->format() == ops[1].getValue()->format();
}

// Uniform (scalar) conditional branch on SCC — the idiom for uniform loops/ifs. entryID (small set).
static bool isSccBranch(const Instruction &in){
  const entryID id = in.getOperation().getID();
  return id == amdgpu_gfx908_op_S_CBRANCH_SCC0 || id == amdgpu_gfx908_op_S_CBRANCH_SCC1;
}
static bool sccBranchTakenWhenTrue(const Instruction &in){   // SCC1 branches when the condition holds
  return in.getOperation().getID() == amdgpu_gfx908_op_S_CBRANCH_SCC1;
}

// Divergent (EXEC-mask) conditional branch — the backedge of an EXEC loop / waterfall. entryID.
static bool isExecBranch(const Instruction &in){
  const entryID id = in.getOperation().getID();
  return id == amdgpu_gfx908_op_S_CBRANCH_EXECNZ || id == amdgpu_gfx908_op_S_CBRANCH_EXECZ;
}
static bool execBranchTakenWhenLanesRemain(const Instruction &in){   // EXECNZ branches while any lane is active
  return in.getOperation().getID() == amdgpu_gfx908_op_S_CBRANCH_EXECNZ;
}
// Cross-lane broadcast (v_readfirstlane/v_readlane) — uniformizes a divergent value. entryID.
static bool isReadfirstlane(const Instruction &in){
  const entryID id = in.getOperation().getID();
  return id == amdgpu_gfx908_op_V_READFIRSTLANE_B32 || id == amdgpu_gfx908_op_V_READLANE_B32;
}
// Indirect call / jump-through-register (the target is a register operand). entryID.
static bool isCallInsn(const Instruction &in){
  const entryID id = in.getOperation().getID();
  return id == amdgpu_gfx908_op_S_SWAPPC_B64 || id == amdgpu_gfx908_op_S_SETPC_B64;
}

// SKETCH (Dyninst has NO induction-variable analysis — getLoopIterators() is a warning stub — so
// this is a first cut to port into Dyninst/ParseAPI and upstream). A BASIC induction variable is an
// additive self-recurrence `R = R + <const>`: the destination register is also read, and another
// source is a constant. The constant is detected via Expression::eval() (a defined Result means a
// compile-time constant), NOT operand text — so negative inline constants like `-1` are handled.
static bool isBasicInductionUpdate(const Instruction &in, std::string &reg, std::string &step){
  if(in.getOperation().format().find("ADD") == std::string::npos) return false;  // additive only (sketch)
  std::vector<Operand> ops; in.getOperands(ops);
  if(ops.size() < 2 || !ops[0].isWritten()) return false;
  const std::string dst = ops[0].getValue()->format();
  if(dst.empty()) return false;
  bool readsSelf = false, haveImm = false; std::string imm;
  for(size_t i = 1; i < ops.size(); ++i){
    if(!ops[i].isRead()) continue;
    const Result r = ops[i].getValue()->eval();
    if(r.defined){                                                 // a compile-time constant operand
      long long v = r.convert<long long>();
      if(v > 0x7fffffffLL && v <= 0xffffffffLL) v -= 0x100000000LL;  // interpret 32-bit steps as signed
      imm = std::to_string(v); haveImm = true;
    } else if(ops[i].getValue()->format() == dst) readsSelf = true;  // reads the destination register
  }
  if(readsSelf && haveImm){ reg = dst; step = imm; return true; }
  return false;
}

// Semantic classification by opcode identity (entryID); for the context-dependent scalar
// ALU ops, the EXEC write-set decides whether they act on the mask or are plain arithmetic.
static MaskOp classify(const Instruction &in){
  switch(in.getOperation().getID()){
    // OPEN: narrows EXEC by a fresh condition (s_and_saveexec) — a divergent `if`.
    case amdgpu_gfx908_op_S_AND_SAVEEXEC_B64:
      return MaskOp::Open;

    // ELSE: any saveexec that isn't the plain AND-narrow is a same-region transition — else-entry
    // widen (s_or_saveexec), flip (s_xor_saveexec), or the rarer boolean-combine forms the exec-mask
    // fuser (SIOptimizeExecMasking::getSaveExecOp) can emit for nand/nor/orn2/xnor conditions.
    case amdgpu_gfx908_op_S_OR_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_XOR_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_ORN2_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_NAND_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_NOR_SAVEEXEC_B64:
    case amdgpu_gfx908_op_S_XNOR_SAVEEXEC_B64:
      return MaskOp::Else;
    // s_andn2_saveexec: same-reg (s[X],s[X]) is a clean else flip; different regs (s[X],s[Y])
    // route the REMAINING lanes to the next case — an elseif-chain step via a precomputed mask,
    // which does NOT map onto a plain else (forcing it produced double-negated garbage).
    case amdgpu_gfx908_op_S_ANDN2_SAVEEXEC_B64:
      return isSelfFlip(in) ? MaskOp::Else : MaskOp::Reroute;

    // Context-dependent: a CLOSE/ELSE only when it actually writes EXEC (restore / flip);
    // otherwise it is ordinary scalar arithmetic on SGPRs.
    case amdgpu_gfx908_op_S_OR_B64:
    case amdgpu_gfx908_op_S_MOV_B64:
      return writesExec(in) ? MaskOp::Close : MaskOp::None;
    case amdgpu_gfx908_op_S_XOR_B64:
    case amdgpu_gfx908_op_S_ANDN2_B64:
      return writesExec(in) ? MaskOp::Else : MaskOp::None;

    // NARROW: `exec &= cond` inside a region — the short-circuit `&&` refinement
    // (`if (A && B)` lowers to saveexec(A) then s_and_b64 exec,exec,B).
    case amdgpu_gfx908_op_S_AND_B64:
      return writesExec(in) ? MaskOp::Narrow : MaskOp::None;

    case amdgpu_gfx908_op_V_CNDMASK_B32: return MaskOp::Select;   // predicated value-select
    case amdgpu_gfx908_op_S_SWAPPC_B64:  return MaskOp::Call;     // indirect call (region body)

    default: {
      // Compares are the deliberate exception to the entryID rule: ~200 variants (V_CMP/S_CMP/
      // S_CMPK × conds × types), no InsnCategory, and the operator is per-opcode — so we key on the
      // opcode-only mnemonic. Bonus: mnemonics are identical across gfx908/90a/940, so this is
      // arch-portable for free (unlike per-arch entryIDs). v_cmp → VCC/SGPR mask; s_cmp/k → SCC.
      const std::string op = in.getOperation().format();
      if(op.rfind("V_CMP", 0) == 0 || op.rfind("S_CMP", 0) == 0) return MaskOp::Compare;
      return MaskOp::None;
    }
  }
}

// ---------- semantic condition recovery ---------------------------------------------

// Map a V_CMP_<COND>_<TYPE> opcode mnemonic to its infix comparison operator.
static std::string cmpOp(const std::string &mnemonic){
  // Split on '_' and take the condition field, e.g. "V_CMP_LT_F32" -> "LT".
  std::vector<std::string> parts;
  std::stringstream ss(mnemonic);
  for(std::string part; std::getline(ss, part, '_'); ) parts.push_back(part);
  const std::string cond = parts.size() > 2 ? parts[2] : "?";

  if(cond == "LT")                     return "<";
  if(cond == "GT")                     return ">";
  if(cond == "LE" || cond == "NGT")    return "<=";
  if(cond == "GE" || cond == "NLT")    return ">=";
  if(cond == "EQ")                     return "==";
  if(cond == "NE" || cond == "LG" || cond == "NEQ") return "!=";
  return "." + cond + ".";             // unmapped condition — surfaced, not hidden
}

// The registers an instruction reads as its condition input, excluding EXEC (an implicit
// read of the saveexec/restore ops, not part of the condition).
static std::set<MachRegister> maskReads(const Instruction &in){
  std::set<RegisterAST::Ptr> reads;
  in.getReadSet(reads);
  std::set<MachRegister> out;
  for(const auto &r : reads){
    MachRegister id = r->getID();
    if(id == R::exec_lo || id == R::exec_hi) continue;
    out.insert(id);
  }
  return out;
}

// True if the instruction writes any register in `mask` (identity match on the write set).
static bool definesAny(const Instruction &in, const std::set<MachRegister> &mask){
  std::set<RegisterAST::Ptr> written;
  in.getWriteSet(written);
  for(const auto &w : written) if(mask.count(w->getID())) return true;
  return false;
}
// True if the instruction writes a register whose display name is `name`.
static bool definesName(const Instruction &in, const std::string &name){
  std::set<RegisterAST::Ptr> written;
  in.getWriteSet(written);
  for(const auto &w : written) if(w->format() == name) return true;
  return false;
}
// The read operands (registers + immediates) of an instruction, excluding flag registers.
static std::vector<std::string> readOperands(const Instruction &in){
  std::vector<std::string> out;
  std::vector<Operand> ops; in.getOperands(ops);
  for(const auto &o : ops){
    if(!o.isRead() || o.isWritten()) continue;
    const std::string s = o.getValue()->format();
    if(s=="EXEC"||s=="EXEC_LO"||s=="EXEC_HI"||s=="SCC"||s=="VCC"||s=="[VCC_LO,VCC_HI]") continue;
    out.push_back(s);
  }
  return out;
}

// ---------- CFRecovery API ----------------------------------------------------------
namespace CFR {
enum class Kind { Root, If, Else, Loop, Select };
struct Region {
  Kind kind=Kind::Root; std::string cond; Address header=0; Block* headerBlk=nullptr;
  Region* parent=nullptr; std::vector<Region*> children; std::vector<Address> body;
  Address reconverge=0; bool balanced=true;
  Region* pairedIf=nullptr;   // for an Else: the If it is the `else` of (its guard is !pairedIf)
  std::string effCond;        // effective guard: AND of local guards from root down (cross-tree)
};
class StructureAnalysis {
  Function* f; SymtabAPI::Symtab* symtab_; std::vector<Region*> all_; Region* root_;
  std::vector<std::pair<Address,Instruction>> ins;      // address-ordered
  std::map<Address,Block*> blkOf;
  std::map<Address,std::string> condAt;   // forward-provenance-substituted condition, per compare addr
  std::map<uint64_t,std::string> kernargNames_;   // kernarg offset → arg name (implicit + explicit, from notes)
  std::map<Address,std::string> callTargetAt_;    // indirect-call site → provenance of its target register
  struct PackedSrc { std::string base; long long off; };   // a reg loaded from base+off (kernarg/dispatch)
  std::map<std::string,PackedSrc> packed_;        // reg → its (base,offset), for packed-lane (>>16) extraction
 public:
  explicit StructureAnalysis(Function* fn, SymtabAPI::Symtab* st = nullptr):f(fn),symtab_(st){ build(); }
  Region* root() const { return root_; }
  // Condition of the OPEN/NARROW at instruction index `i`: find the reaching definition of its
  // mask register(s) by walking control-flow BACKWARD (this block, then predecessor edges) and
  // format it — NOT by scanning the flat instruction vector, which crosses unrelated paths.
  std::string recoverCond(int i){
    const std::set<MachRegister> mask = maskReads(ins[i].second);
    std::set<Block*> visited;
    const std::string c = reachingCond(mask, blkOf[ins[i].first], ins[i].first, visited);
    return c.empty() ? "?" : c;
  }

  // Nearest reaching definition of `mask`, control-flow-backward: scan `blk`'s instructions
  // before `before`, then recurse through predecessor edges. Returns the formatted condition
  // (a V_CMP → infix comparison; any other producer surfaced by opcode — mask-algebra recursion
  // is a TODO), or "" if none reaches. First def along the DFS path; a per-path merge/phi is not
  // modeled (fine when the condition is computed in one dominating predecessor, the common case).
  std::string reachingCond(const std::set<MachRegister> &mask, Block *blk, Address before,
                           std::set<Block*> &visited){
    if(!blk || visited.count(blk)) return "";
    visited.insert(blk);

    Block::Insns insns; blk->getInsns(insns);
    for(auto it = insns.rbegin(); it != insns.rend(); ++it){
      if(it->first >= before) continue;              // at/after the use — not a prior def
      if(definesAny(it->second, mask)){
        if(classify(it->second) == MaskOp::Compare){
          auto pit = condAt.find(it->first);         // forward-provenance-substituted (roots in implicit args)
          return (pit != condAt.end() && !pit->second.empty()) ? pit->second : formatCompare(it->second);
        }
        return it->second.getOperation().format();
      }
    }
    for(Edge *e : blk->sources()){                   // in-edges: predecessor blocks
      if(!intraEdge(e->type())) continue;
      Block *pred = e->src();
      if(pred){
        const std::string c = reachingCond(mask, pred, pred->end(), visited);
        if(!c.empty()) return c;
      }
    }
    return "";
  }

  // Reaching-def INSTRUCTION of `regs` (control-flow backward). `stayOutOf` skips a loop's blocks —
  // used to find a value's INIT in the preheader rather than its in-loop update.
  bool reachingDefInsn(const std::set<MachRegister> &regs, Block *blk, Address before,
                       std::set<Block*> &visited, Instruction &out, Address &defAddr, Loop *stayOutOf = nullptr){
    if(!blk || visited.count(blk) || (stayOutOf && stayOutOf->hasBlock(blk))) return false;
    visited.insert(blk);
    Block::Insns insns; blk->getInsns(insns);
    for(auto it = insns.rbegin(); it != insns.rend(); ++it){
      if(it->first >= before) continue;
      if(definesAny(it->second, regs)){ out = it->second; defAddr = it->first; return true; }
    }
    for(Edge *e : blk->sources())
      if(intraEdge(e->type()) && e->src() &&
         reachingDefInsn(regs, e->src(), e->src()->end(), visited, out, defAddr, stayOutOf)) return true;
    return false;
  }
  // Same, matching a register by display name (used to find the loop counter's init).
  bool reachingDefByName(const std::string &name, Block *blk, Address before,
                         std::set<Block*> &visited, Instruction &out, Loop *stayOutOf = nullptr){
    if(!blk || visited.count(blk) || (stayOutOf && stayOutOf->hasBlock(blk))) return false;
    visited.insert(blk);
    Block::Insns insns; blk->getInsns(insns);
    for(auto it = insns.rbegin(); it != insns.rend(); ++it){
      if(it->first >= before) continue;
      if(definesName(it->second, name)){ out = it->second; return true; }
    }
    for(Edge *e : blk->sources())
      if(intraEdge(e->type()) && e->src() &&
         reachingDefByName(name, e->src(), e->src()->end(), visited, out, stayOutOf)) return true;
    return false;
  }

  // Format a V_CMP as "(srcA <op> srcB)" from its source operands (skipping the implicit
  // mask/flag registers), or fall back to the opcode if the operands aren't recognizable.
  std::string formatCompare(const Instruction &cmp){
    std::vector<Operand> operands;
    cmp.getOperands(operands);

    std::vector<std::string> src;
    for(const auto &op : operands){
      if(!op.isRead() || op.isWritten()) continue;
      const std::string s = op.getValue()->format();
      if(s == "EXEC" || s == "SCC" || s == "VCC" || s == "[VCC_LO,VCC_HI]") continue;
      src.push_back(s);
    }

    if(src.size() >= 2)
      return "(" + src[0] + " " + cmpOp(cmp.getOperation().format()) + " " + src[1] + ")";
    return cmp.getOperation().format();
  }
  Region* regionOf(Block* b){                      // which construct owns a block
    Address a=b->start(); Region* best=root_;
    for(auto*r:all_) if(r!=root_ && r->header<=a && (r->reconverge==0||a<r->reconverge))
      if(!best||r->header>=best->header) best=r;
    return best;
  }

  // Uniform (scalar) loops. Unlike EXEC divergence, s_cbranch_scc are real CFG branches, so
  // ParseAPI's getLoops() already has the backedge structure — we just recover each loop's
  // condition from the SCC-setting compare feeding its latch s_cbranch_scc, and whether the taken
  // edge stays in the loop (continue) or exits. CFG-based, independent of the EXEC region tree.
  void recoverScalarLoops(){
    std::vector<Loop*> loops;
    f->getLoops(loops);
    for(Loop *L : loops){
      std::vector<Block*> entries; L->getLoopEntries(entries);
      const Address hdr = entries.empty() ? 0 : entries[0]->start();
      std::vector<Block*> blocks; L->getLoopBasicBlocks(blocks);

      // 1) collect basic induction variables in the loop: reg name -> step.
      std::map<std::string,std::string> ivs;
      for(Block *b : blocks){
        Block::Insns insns; b->getInsns(insns);
        for(const auto &kv : insns){
          std::string reg, step;
          if(isBasicInductionUpdate(kv.second, reg, step)) ivs.emplace(reg, step);
        }
      }

      // 2) each latch s_cbranch_scc: recover the condition + the PRIMARY counter (the IV that the
      //    condition tests), its bound (the other compare operand), init (preheader), and step.
      for(Block *b : blocks){
        Block::Insns insns; b->getInsns(insns);
        if(insns.empty()) continue;
        const Address     termAddr = insns.rbegin()->first;
        const Instruction term     = insns.rbegin()->second;
        if(!isSccBranch(term)) continue;

        std::set<MachRegister> scc{ R::src_scc };
        std::set<Block*> vis; Instruction cmp; Address cmpAddr = 0;
        const bool haveCmp = reachingDefInsn(scc, b, termAddr, vis, cmp, cmpAddr);
        std::string cond;
        if(haveCmp){
          auto pit = condAt.find(cmpAddr);           // forward-provenance-substituted, if available
          cond = (pit != condAt.end() && !pit->second.empty()) ? pit->second
               : (classify(cmp) == MaskOp::Compare ? formatCompare(cmp) : cmp.getOperation().format());
        }

        bool takenInLoop = false;
        for(Edge *e : b->targets())
          if(e->type() == COND_TAKEN){ takenInLoop = e->trg() && L->hasBlock(e->trg()); break; }
        const bool whenTrue = sccBranchTakenWhenTrue(term);
        printf("  loop hdr@0x%lx  latch@0x%lx  %s %s%s\n", (unsigned long)hdr, (unsigned long)termAddr,
               takenInLoop ? "continue-if" : "exit-if", whenTrue ? "" : "!", cond.empty() ? "(?)" : cond.c_str());

        if(!haveCmp) continue;
        std::string counter, bound;                          // counter = a compare operand that's an IV
        for(const std::string &c : readOperands(cmp)){ if(ivs.count(c)) counter = c; else bound = c; }
        if(counter.empty()) continue;

        std::string init = "?";                              // init = def of the counter in the preheader
        for(Block *entry : entries){
          for(Edge *e : entry->sources()){
            if(!intraEdge(e->type())) continue;
            Block *p = e->src();
            if(!p || L->hasBlock(p)) continue;               // skip the in-loop backedge predecessor
            std::set<Block*> v2; Instruction idef;
            if(reachingDefByName(counter, p, p->end(), v2, idef, L)){ init = idef.format(); break; }
          }
          if(init != "?") break;
        }
        printf("    counter %s : init [%s]  bound %s  step %s\n", counter.c_str(),
               init.c_str(), bound.empty() ? "?" : bound.c_str(), ivs[counter].c_str());
      }
    }
  }

  // Divergent (EXEC-mask) loops. Unlike a scalar loop, the backedge is s_cbranch_execnz/execz (repeat
  // WHILE lanes remain active), not an SCC compare — so recoverScalarLoops skips it. The important
  // instance is the WATERFALL that lowers an indirect call / jump-table dispatch: v_readfirstlane picks
  // one active lane's target, v_cmp_eq(x, firstlane(x)) + s_and_saveexec masks the lanes sharing it, the
  // call runs under that mask, and execnz loops for the remaining distinct targets. CFG-based (getLoops).
  void recoverExecLoops(){
    std::vector<Loop*> loops;
    f->getLoops(loops);
    for(Loop *L : loops){
      std::vector<Block*> blocks; L->getLoopBasicBlocks(blocks);
      std::vector<Block*> entries; L->getLoopEntries(entries);
      const Address hdr = entries.empty() ? 0 : entries[0]->start();

      // latch = a loop block terminated by an EXEC-conditional branch (else it's a scalar loop / not one)
      Address latch = 0; bool whileLanes = true; bool isExec = false;
      for(Block *b : blocks){
        Block::Insns insns; b->getInsns(insns);
        if(insns.empty()) continue;
        const Instruction term = insns.rbegin()->second;
        if(isExecBranch(term)){ latch = insns.rbegin()->first; whileLanes = execBranchTakenWhenLanesRemain(term); isExec = true; break; }
      }
      if(!isExec) continue;

      // waterfall signature (a cross-lane broadcast in the body) + its per-iteration selector — the
      // convergence compare, which computeProvenance already rendered as `(firstlane(x) == x)`. Prefer
      // that exact convergence form over any other firstlane-bearing compare in the loop.
      bool waterfall = false; std::string selector, fallback;
      for(Block *b : blocks){
        Block::Insns insns; b->getInsns(insns);
        for(const auto &kv : insns){
          if(isReadfirstlane(kv.second)) waterfall = true;
          auto pit = condAt.find(kv.first);
          if(pit == condAt.end()) continue;
          const std::string &c = pit->second;
          const size_t fl = c.find("firstlane(");
          if(fl == std::string::npos) continue;
          if(fallback.empty()) fallback = c;
          const size_t close = c.find(')', fl + 10), eq = c.find(" == ", close);   // firstlane(X) == X ?
          if(close != std::string::npos && eq != std::string::npos){
            std::string inner = c.substr(fl + 10, close - (fl + 10));
            std::string rhs   = c.substr(eq + 4); if(!rhs.empty() && rhs.back() == ')') rhs.pop_back();
            if(inner == rhs) selector = c;                                          // the true convergence test
          }
        }
      }
      if(selector.empty()) selector = fallback;

      // Does a call in this loop actually jump THROUGH a uniformized (firstlane) value? That def-use —
      // not the waterfall shape — is what makes it an indirect-call dispatch. Otherwise the waterfall is
      // just serializing a divergent value for some other use (e.g. a hostcall payload, a scalar load).
      std::string dispatchTarget;
      for(Block *b : blocks){
        Block::Insns insns; b->getInsns(insns);
        for(const auto &kv : insns){
          auto it = callTargetAt_.find(kv.first);
          if(it != callTargetAt_.end() && it->second.find("firstlane(") != std::string::npos) dispatchTarget = it->second;
        }
      }

      std::string kind, extra;
      if(!dispatchTarget.empty()){ kind = "indirect-call dispatch"; extra = "  target " + dispatchTarget; }
      else if(waterfall){ kind = "waterfall serialization"; extra = selector.empty() ? "" : ("  groups by " + selector); }
      else kind = "divergent loop";
      printf("  exec-loop hdr@0x%lx  latch@0x%lx  %s  [%s]%s\n",
             (unsigned long)hdr, (unsigned long)latch,
             whileLanes ? "repeat-while-lanes-remain" : "repeat-until-lanes-clear",
             kind.c_str(), extra.c_str());
    }
  }
 private:
  // Allocate a region and register it in all_ (which owns every node for cleanup).
  Region* makeRegion(Kind kind, Address header, Block* headerBlk){
    Region* r = new Region();
    r->kind = kind; r->header = header; r->headerBlk = headerBlk;
    all_.push_back(r);
    return r;
  }

  // Follow only intraprocedural edges — stay inside this function, don't descend into callees.
  static bool intraEdge(EdgeTypeEnum t){
    return t == COND_TAKEN || t == COND_NOT_TAKEN || t == DIRECT ||
           t == FALLTHROUGH || t == CALL_FT || t == INDIRECT;
  }

  // Reverse post-order of the CFG from the entry block: each block is visited before its
  // successors (except across backedges), so a region's OPEN precedes its body precedes its
  // reconvergence — the order the stack machine below requires. Address order does NOT
  // guarantee this once the compiler reorders blocks or emits backward branches.
  std::vector<Block*> reversePostOrder(){
    std::set<Block*> mine;
    for(Block* b : f->blocks()) mine.insert(b);

    std::vector<Block*> post;
    std::set<Block*> seen;
    std::vector<std::pair<Block*, bool>> work{ {f->entry(), false} };
    while(!work.empty()){
      Block* b = work.back().first;
      bool expanded = work.back().second;
      work.pop_back();
      if(expanded){ post.push_back(b); continue; }   // children done -> emit in post-order
      if(seen.count(b)) continue;
      seen.insert(b);
      work.push_back({b, true});
      for(Edge* e : b->targets()){
        if(!intraEdge(e->type())) continue;
        Block* t = e->trg();
        if(t && mine.count(t) && !seen.count(t)) work.push_back({t, false});
      }
    }
    std::reverse(post.begin(), post.end());
    return post;
  }

  void build(){
    root_ = makeRegion(Kind::Root, /*header=*/0, /*headerBlk=*/nullptr);

    // Collect instructions in CONTROL-FLOW order (reverse post-order of the CFG from the entry
    // block), NOT address order — the compiler reorders blocks and emits backward branches, so
    // address proximity does not reflect execution/nesting. blkOf records the block each
    // instruction lives in (for the post-dominance reconvergence query in openRegion).
    for(Block* b : reversePostOrder()){
      Block::Insns insns; b->getInsns(insns);
      for(const auto& kv : insns){ ins.push_back(kv); blkOf[kv.first] = b; }
    }

    // Forward value-provenance runs FIRST (separate pass over the finished CFG, see design notes) so
    // condAt[] is populated before conditions are recovered — lets EXEC if-conditions and loop
    // conditions alike root their operands in the implicit args (threadIdx/blockIdx/kernarg).
    computeProvenance();

    // Walk instructions in control-flow order, maintaining a stack of open regions. classify()
    // drives the transitions; the stack depth is the nesting depth.
    std::vector<Region*> stack{root_};
    for(int i = 0; i < (int)ins.size(); ++i){
      const Address addr = ins[i].first;
      Block* const blk   = blkOf[addr];
      switch(classify(ins[i].second)){
        case MaskOp::Open:    openRegion(i, addr, blk, stack);    break;
        case MaskOp::Else:    elseRegion(addr, blk, stack);       break;
        case MaskOp::Reroute: rerouteRegion(addr, blk, stack);   break;
        case MaskOp::Close:   closeRegion(addr, stack);           break;
        case MaskOp::Narrow: narrowRegion(i, stack);             break;
        case MaskOp::Call:
        case MaskOp::Select: stack.back()->body.push_back(addr); break;
        default: break;
      }
    }

    // Cross-tree condition combination: fold each region's local guard into the full guard
    // under which it executes (AND of guards from root down).
    combineConditions(root_, "");

  }

  // The guard local to a region relative to its parent: an If's own condition, or the negation
  // of the sibling If it is the `else` of.
  std::string localGuard(Region *r){
    if(r->kind == Kind::Else && r->pairedIf) return "!(" + localGuard(r->pairedIf) + ")";
    return r->cond;
  }

  // Fill effCond for the subtree: effective guard = parent's effective guard AND this region's
  // local guard. (`if(A){ if(B) }` gives the inner guard `A && B`; an else gives `... && !A`.)
  void combineConditions(Region *r, const std::string &parentEff){
    const std::string self = localGuard(r);
    r->effCond = self.empty()      ? parentEff
               : parentEff.empty() ? self
               :                     parentEff + " && " + self;
    for(Region *c : r->children) combineConditions(c, r->effCond);
  }

  // ---- Forward value-provenance (separate pass over the finished CFG) ----------------------------
  // Seeds ABI meanings at entry and flows them forward with KILL-ON-OVERWRITE, so a value's meaning
  // is position-correct even when the kernel reuses the kernarg-pointer / ABI registers (which
  // backward slicing can't soundly attribute). v1: seeds threadIdx.x (v0) + the kernarg pointer
  // (first live-in S_LOAD base); transfer for mov/load/add/mul/lshl; snapshots the substituted
  // condition at each compare into condAt. Single RPO pass (no loop fixpoint yet).
  // Entry ABI seeds, KD-derived: reuses computeAbiSgprLayout's placement logic, read straight from the
  // 64-byte descriptor — NO KD object constructed (analysis only reads). Falls back to v0=threadIdx.x.
  std::map<std::string,std::string> abiSeeds(){
    std::map<std::string,std::string> seeds;
    seeds["V0"] = "threadIdx.x";                                // workitem-id.x is always in v0
    if(!symtab_ || !f) return seeds;
    std::vector<SymtabAPI::Symbol*> syms;
    if(!symtab_->findSymbol(syms, f->name() + ".kd") || syms.empty()) return seeds;
    SymtabAPI::Region *rgn = symtab_->findEnclosingRegion(syms[0]->getOffset());
    if(!rgn || !rgn->getPtrToRawData()) return seeds;
    const unsigned char *kd = (const unsigned char*)rgn->getPtrToRawData()
                            + (syms[0]->getOffset() - rgn->getMemOffset());
    uint32_t rsrc2; std::memcpy(&rsrc2, kd + 0x34, 4);          // COMPUTE_PGM_RSRC2
    uint16_t props; std::memcpy(&props, kd + 0x38, 2);          // kernel_code_properties

    // USER SGPRs packed from s0 by the code_properties enables; we only need the kernarg pointer.
    int s = 0; auto user = [&](int bit, int n){ if(props & (1<<bit)){ int at=s; s+=n; return at; } return -1; };
    user(0,4);                                                 // private_segment_buffer
    const int dispatchPtr = user(1,2);                         // dispatch_ptr → hsa_kernel_dispatch_packet
    user(2,2);                                                 // queue_ptr
    const int kernargPtr = user(3,2);                          // kernarg_segment_ptr
    // SYSTEM SGPRs from user_sgpr_count (RSRC2[5:1]); wgid x/y/z = RSRC2 bits 7/8/9.
    int sys = (rsrc2 >> 1) & 0x1f;
    auto wg = [&](int bit){ if(rsrc2 & (1<<bit)){ int at=sys; sys+=1; return at; } return -1; };
    const int wgX = wg(7), wgY = wg(8), wgZ = wg(9);
    const int widDims = (rsrc2 >> 11) & 0x3;                    // ENABLE_VGPR_WORKITEM_ID

    // pair register keys must match InstructionAPI format(): "[S4,S5]" (not "S[4:5]")
    auto sp = [](int lo){ return "[S" + std::to_string(lo) + ",S" + std::to_string(lo+1) + "]"; };
    if(kernargPtr >= 0)  seeds[sp(kernargPtr)]  = "&kernarg";
    if(dispatchPtr >= 0) seeds[sp(dispatchPtr)] = "&dispatch";   // mechanism A: KD-alone blockDim/gridSize
    if(wgX >= 0) seeds["S" + std::to_string(wgX)] = "blockIdx.x";
    if(wgY >= 0) seeds["S" + std::to_string(wgY)] = "blockIdx.y";
    if(wgZ >= 0) seeds["S" + std::to_string(wgZ)] = "blockIdx.z";
    if(widDims >= 1) seeds["V1"] = "threadIdx.y";
    if(widDims >= 2) seeds["V2"] = "threadIdx.z";
    return seeds;
  }

  // This function's kernarg segment as offset → name, from the AMDGPU metadata note (msgpack):
  // HIDDEN implicit args (blockDim/gridDim/… — mechanism B) AND EXPLICIT args by their source name
  // (`.name`). The compiler assigns these offsets (only in the notes — the KD can't locate them), so
  // a load off &kernarg at one of these offsets is a read of that named argument.
  std::map<uint64_t,std::string> kernargNameMap(){
    std::map<uint64_t,std::string> out;
    if(!symtab_ || !f) return out;
    SymtabAPI::Region *note = nullptr;
    if(!symtab_->findRegion(note, ".note") || !note || !note->getPtrToRawData()) return out;

    // Scan ELF notes for NT_AMDGPU_METADATA (type 32, name "AMDGPU"); its desc is the msgpack blob.
    const unsigned char *p = (const unsigned char*)note->getPtrToRawData();
    const unsigned char *end = p + note->getDiskSize();
    const unsigned char *meta = nullptr; size_t metalen = 0;
    while(p + 12 <= end){
      uint32_t namesz, descsz, ntype;
      std::memcpy(&namesz, p, 4); std::memcpy(&descsz, p+4, 4); std::memcpy(&ntype, p+8, 4);
      const unsigned char *nm = p + 12;
      const unsigned char *desc = nm + ((namesz + 3) & ~3u);
      if(desc > end) break;
      if(ntype == 32 && namesz >= 6 && std::memcmp(nm, "AMDGPU", 6) == 0){ meta = desc; metalen = descsz; break; }
      p = desc + ((descsz + 3) & ~3u);
    }
    if(!meta) return out;

    MsgVal root;
    if(!mpParse(meta, meta + metalen, root)) return out;
    const MsgVal *kernels = mpGet(root, "amdhsa.kernels");
    if(!kernels || kernels->t != MsgVal::ARR) return out;

    // The note keys kernels by MANGLED name (`.symbol` = "<mangled>.kd"), but f->name() is demangled —
    // so match against the KD symbol's own mangled name, not f->name().
    std::string kdMangled;
    std::vector<SymtabAPI::Symbol*> ks;
    if(symtab_->findSymbol(ks, f->name() + ".kd") && !ks.empty()) kdMangled = ks[0]->getMangledName();

    for(const MsgVal &k : kernels->arr){
      const MsgVal *sym = mpGet(k, ".symbol");                 // "<mangled>.kd"
      const MsgVal *nm  = mpGet(k, ".name");                   // "<mangled>"
      const bool mine = (sym && sym->t==MsgVal::STR && sym->s == kdMangled)
                     || (nm  && nm->t==MsgVal::STR  && nm->s + ".kd" == kdMangled);
      if(!mine) continue;
      const MsgVal *argsp = mpGet(k, ".args");
      if(!argsp || argsp->t != MsgVal::ARR) break;
      int argIdx = 0;                                           // position among EXPLICIT args (= source param #)
      for(const MsgVal &a : argsp->arr){
        const MsgVal *vk = mpGet(a, ".value_kind");
        const MsgVal *of = mpGet(a, ".offset");
        if(!vk || vk->t!=MsgVal::STR || !of || of->t!=MsgVal::INT) continue;
        std::string name = prettyHidden(vk->s);                 // implicit (blockDim/gridDim/…)
        if(name.empty()){                                       // explicit arg: source `.name`, else positional
          const MsgVal *an = mpGet(a, ".name");                 // present on -g / older-ROCm builds
          name = (an && an->t==MsgVal::STR && !an->s.empty()) ? an->s : ("arg" + std::to_string(argIdx));
          const std::string co = ".coerce";                     // HIP names coerced pointer args "<p>.coerce"
          if(name.size() > co.size() && name.compare(name.size()-co.size(), co.size(), co) == 0)
            name.resize(name.size() - co.size());
          ++argIdx;
        }
        out[(uint64_t)of->i] = name;
      }
      break;
    }
    return out;
  }

  void computeProvenance(){
    std::map<std::string,std::string> state = abiSeeds();       // ABI entry meanings (KD-derived)
    kernargNames_ = kernargNameMap();                          // kernarg offsets → names (implicit + explicit)
    if(std::getenv("CFR_DUMP_ARGS"))                            // diagnostic: show the parsed kernarg layout
      for(const auto &kv : kernargNames_)
        fprintf(stderr, "  [arg] kernarg+0x%-3llx = %s\n", (unsigned long long)kv.first, kv.second.c_str());
    std::string kernargBase;
    for(const auto &kv : state) if(kv.second == "&kernarg") kernargBase = kv.first;   // exact kernarg-ptr reg
    packed_.clear();
    for(Block *b : reversePostOrder()){
      Block::Insns insns; b->getInsns(insns);
      for(const auto &kv : insns){
        if(classify(kv.second) == MaskOp::Compare) condAt[kv.first] = substituted(kv.second, state);
        if(isCallInsn(kv.second)) callTargetAt_[kv.first] = callTarget(kv.second, state);   // its target's provenance
        applyTransfer(kv.second, state, kernargBase);
      }
    }
  }

  // The provenance of an indirect call's TARGET register (the SGPR it jumps through) — this is what
  // tells "is the uniformized value actually used as a call target" vs merely the waterfall shape.
  std::string callTarget(const Instruction &in, const std::map<std::string,std::string> &state){
    std::vector<Operand> ops; in.getOperands(ops);
    for(const auto &o : ops){
      if(o.isWritten()) continue;                                // skip the written return-address reg
      std::set<RegisterAST::Ptr> rs; o.getReadSet(rs);
      const std::vector<MachRegister> rr = dataRegs(rs);
      if(rr.empty() || !isScalarReg(rr.front())) continue;       // the target is a scalar (SGPR pair)
      const std::string key = o.getValue()->format();
      auto it = state.find(key); return it != state.end() ? it->second : key;
    }
    return "";
  }

  // Format a compare with its operands replaced by their current symbolic provenance.
  std::string substituted(const Instruction &cmp, const std::map<std::string,std::string> &state){
    std::vector<std::string> src = readOperands(cmp);
    for(std::string &s : src){ auto it = state.find(s); if(it != state.end()) s = it->second; }
    if(src.size() >= 2) return "(" + src[0] + " " + cmpOp(cmp.getOperation().format()) + " " + src[1] + ")";
    return "";
  }

  // Name of the field at `base`+`off`, where base is "&kernarg" (notes) or "&dispatch" (packet). "" if none.
  std::string labelAt(const std::string &base, long long off){
    if(base == "&dispatch") return packetField(off);
    auto h = kernargNames_.find((uint64_t)off);
    return h != kernargNames_.end() ? h->second : "";
  }

  // Transfer function: update `state` for one instruction (kill-on-write is the default). Operand
  // decomposition/classification is by register IDENTITY (getReadSet/getWriteSet → MachRegister), not
  // by parsing Operand::format(): each source carries its low-half key + scalar flag from the reg set.
  void applyTransfer(const Instruction &in, std::map<std::string,std::string> &state, std::string &kernargBase){
    std::vector<Operand> ops; in.getOperands(ops);
    if(ops.empty() || !ops[0].isWritten()) return;
    const std::string opc = in.getOperation().format();

    std::set<RegisterAST::Ptr> dstSet; ops[0].getWriteSet(dstSet);
    const std::vector<MachRegister> dstRegs = dataRegs(dstSet);
    if(dstRegs.empty()) return;                          // writes only a mask/special reg — not tracked here
    const std::string dst    = ops[0].getValue()->format();
    const std::string dstLow = regKey(dstRegs.front());  // low 32-bit half (== dst for a single register)

    struct Src { std::string key, lowKey; bool scalar; };
    std::vector<Src> src; std::string imm;
    for(size_t i = 1; i < ops.size(); ++i){
      if(ops[i].isWritten()) continue;                   // carry/waitcnt side-writes (VCC, LGKMCNT, …)
      const Result r = ops[i].getValue()->eval();
      if(r.defined){                                     // an immediate — incl. the S_LOAD offset (not isRead)
        if(imm.empty()){ long long v = r.convert<long long>();
          if(v > 0x7fffffffLL && v <= 0xffffffffLL) v -= 0x100000000LL;   // 32-bit signed
          imm = std::to_string(v); }
        continue;
      }
      if(!ops[i].isRead()) continue;
      std::set<RegisterAST::Ptr> rs; ops[i].getReadSet(rs);
      const std::vector<MachRegister> rr = dataRegs(rs);
      if(rr.empty()) continue;                           // side-only operand (m0 / waitcnt) — ignore
      src.push_back({ ops[i].getValue()->format(), regKey(rr.front()), isScalarReg(rr.front()) });
    }
    auto val = [&](const std::string &n){ auto it = state.find(n); return it != state.end() ? it->second : n; };

    if(opc.rfind("S_LOAD", 0) == 0 && !src.empty()){                       // load off a pointer
      const std::string base = src[0].key;
      if(kernargBase.empty() && !state.count(base)) kernargBase = base;    // first live-in base = kernarg ptr
      const std::string meaning = (base == kernargBase) ? "&kernarg" : val(base);   // &kernarg / &dispatch / …
      const long long off = imm.empty() ? 0 : std::strtoll(imm.c_str(), nullptr, 10);
      packed_.erase(dst);
      if(meaning == "&kernarg" || meaning == "&dispatch"){                 // a known segment: label + remember offset
        const std::string lab = labelAt(meaning, off);
        state[dst] = !lab.empty() ? lab : ((meaning == "&kernarg" ? "karg@" : "disp@") + imm);
        packed_[dst] = { meaning, off };                                   // for a later packed-lane (>>16) read
        if(dstLow != dst) state[dstLow] = state[dst];                      // a 64-bit ptr: attribute its low half too
      } else {
        state[dst] = std::string("mem[") + meaning + "]";
      }
      return;
    }
    // VMEM load (global/flat): value = mem[address]. Two addressing forms:
    //   flat  — `global_load vD, v[lo:hi]`          → address is the VGPR pair (low half holds ptr+off)
    //   saddr — `global_load vD, vOff, s[b:b+1] N`  → address is s-base + vOff + N (the s-base is the REAL
    //           pointer; keying only on vOff collapses distinct pointers that reuse an index register).
    if(opc.rfind("GLOBAL_LOAD", 0) == 0 || opc.rfind("FLAT_LOAD", 0) == 0){
      packed_.erase(dst);
      const Src *sbase = nullptr, *voff = nullptr;
      for(const Src &s : src){ if(s.scalar && !sbase) sbase = &s; else if(!voff) voff = &s; }
      if(sbase){                                                          // saddr form: root through the base pointer
        std::string a = val(sbase->key);
        if(voff)                     a += " + " + val(voff->lowKey);
        if(!imm.empty() && imm != "0") a += " + " + imm;
        state[dst] = "mem[" + a + "]";
      } else if(!src.empty()){                                           // flat form: low half of the address pair
        state[dst] = "mem[" + val(src[0].lowKey) + "]";
      } else state.erase(dst);
      return;
    }
    if((opc.rfind("S_MOV", 0) == 0 || opc.rfind("V_MOV", 0) == 0) && !src.empty()){
      auto ps = packed_.find(src[0].key); PackedSrc keep = ps != packed_.end() ? ps->second : PackedSrc{};
      bool carry = ps != packed_.end();
      if(state.count(src[0].key)){ state[dst] = state[src[0].key]; } else state.erase(dst);
      packed_.erase(dst); if(carry) packed_[dst] = keep;                   // copies carry their packed source
      return;
    }
    // Cross-lane broadcast: v_readfirstlane/v_readlane sD, vS = the value of vS in the first active lane
    // (a divergent VGPR → uniform SGPR). Modeling it as firstlane(x) makes the waterfall convergence test
    // `v_cmp_eq(x, firstlane(x))` read as a real test instead of a degenerate `x == x`.
    if(opc.rfind("V_READFIRSTLANE", 0) == 0 || opc.rfind("V_READLANE", 0) == 0){
      packed_.erase(dst);
      if(!src.empty()) state[dst] = "firstlane(" + val(src[0].key) + ")"; else state.erase(dst);
      return;
    }
    // AND with an immediate, two cases distinguished by whether the source is a tracked PACKED field:
    //   packed u16 extract (`s_and sX,sY,0xffff` of a packed dword) → preserve the field label (low half
    //     keeps the same offset; the matching high half is the s_lshr…16 below)
    //   bit-mask / test     (`v_and v6,1,v0` = i%2, or any `x & k`)   → build `(x & imm)` so it roots
    if((opc.rfind("S_AND_B32", 0) == 0 || opc.rfind("V_AND_B32", 0) == 0) && src.size() == 1 && !imm.empty()){
      auto ps = packed_.find(src[0].key);
      const bool packed = ps != packed_.end();
      const PackedSrc keep = packed ? ps->second : PackedSrc{};            // copy out before erase (dst may == src)
      packed_.erase(dst);
      if(packed){                                                          // packed field: preserve label + offset
        if(state.count(src[0].key)) state[dst] = state[src[0].key]; else state.erase(dst);
        packed_[dst] = keep;
      } else if(state.count(src[0].key)){                                  // bit-mask / test: root the value
        state[dst] = "(" + val(src[0].key) + " & " + imm + ")";
      } else state.erase(dst);
      return;
    }
    if(opc.rfind("S_LSHR", 0) == 0 && src.size() == 1 && imm == "16"){
      auto ps = packed_.find(src[0].key);
      const bool has = ps != packed_.end();
      const PackedSrc keep = has ? ps->second : PackedSrc{};              // copy out before erase (dst may == src)
      packed_.erase(dst);
      if(has){
        const long long hi = keep.off + 2;                                // the high u16 lane
        const std::string lab = labelAt(keep.base, hi);
        if(!lab.empty()){ state[dst] = lab; packed_[dst] = { keep.base, hi }; return; }
      }
      state.erase(dst);
      return;
    }
    packed_.erase(dst);                                                    // arithmetic/unknown: not a raw packed field
    // Vector min/max — the compiler lowers `x<W && y<W` to `max(x,y)<W` (and `||` to min); surface it as
    // a function so the combined condition still roots (this is where 2D blockDim.y becomes visible).
    if(opc.rfind("V_MAX", 0) == 0 || opc.rfind("V_MIN", 0) == 0){
      const char *fn = opc.find("MAX") != std::string::npos ? "max" : "min";
      bool known = false; for(const Src &s : src) if(state.count(s.key)) known = true;
      if(known && src.size() >= 2)                       state[dst] = std::string(fn) + "(" + val(src[0].key) + ", " + val(src[1].key) + ")";
      else if(known && src.size() == 1 && !imm.empty()) state[dst] = std::string(fn) + "(" + val(src[0].key) + ", " + imm + ")";
      else state.erase(dst);
      return;
    }
    const bool mul = opc.find("MUL") != std::string::npos, shl = opc.find("LSHL") != std::string::npos;
    if(mul || shl || opc.find("ADD") != std::string::npos){
      const char *op = mul ? " * " : shl ? " << " : " + ";
      // Build the expression; `low` selects each operand's low 32-bit half — the 64-bit address math
      // (lshl_b64/add_co) keeps "pointer + offset" in the low half, which VMEM loads consume.
      auto build = [&](bool low)->std::string {
        auto v = [&](const Src &s){ return val(low ? s.lowKey : s.key); };
        if(src.size() >= 2)               return "(" + v(src[0]) + op + v(src[1]) + ")";
        if(src.size() == 1 && !imm.empty()) return "(" + v(src[0]) + op + imm + ")";
        return "";
      };
      bool known = false; for(const Src &s : src) if(state.count(s.key) || state.count(s.lowKey)) known = true;
      const std::string full = build(false);
      if(known && !full.empty()){
        state[dst] = full;
        if(dstLow != dst){ const std::string lh = build(true); if(!lh.empty()) state[dstLow] = lh; }
      } else { state.erase(dst); if(dstLow != dst) state.erase(dstLow); }
      return;
    }
    state.erase(dst);                                                      // any other write: unknown
  }

  // OPEN (s_and_saveexec): a new `if` nested in the current region. Record its reconvergence
  // as the header block's immediate post-dominator — the structural prediction that the
  // matching close is later checked against.
  void openRegion(int i, Address addr, Block* blk, std::vector<Region*>& stack){
    Region* r = makeRegion(Kind::If, addr, blk);
    r->cond   = recoverCond(i);
    r->parent = stack.back();
    stack.back()->children.push_back(r);
    Block* postdom = blk ? f->getImmediatePostDominator(blk) : nullptr;
    r->reconverge  = postdom ? postdom->start() : 0;
    stack.push_back(r);
  }

  // ELSE (s_andn2_saveexec / s_xor exec): the complementary arm — a sibling under the same
  // parent that replaces the `then` arm on the stack. Exact for if/else; approximate for the
  // elseif chain (which the close's post-dominance cross-check flags as unbalanced).
  void elseRegion(Address addr, Block* blk, std::vector<Region*>& stack){
    Region* cur = stack.back();
    if(cur == root_) return;
    Region* e = makeRegion(Kind::Else, addr, blk);
    e->parent = cur->parent;
    e->pairedIf = cur;                    // this else complements the region it replaces
    if(cur->parent) cur->parent->children.push_back(e);
    stack.back() = e;
  }

  // REROUTE (s_andn2_saveexec s[X],s[Y], Y≠X): the compiler routing the REMAINING lanes to the
  // next case via a precomputed mask — an elseif-chain step that does not map onto a clean else.
  // Create an else arm but do NOT pair it (pairing produced double-negated guards) and flag it
  // irregular, so the output honestly marks "this isn't cleanly structured" instead of guessing.
  void rerouteRegion(Address addr, Block* blk, std::vector<Region*>& stack){
    Region* cur = stack.back();
    if(cur == root_) return;
    Region* e = makeRegion(Kind::Else, addr, blk);
    e->parent = cur->parent;
    e->balanced = false;                  // irregular routing, not a clean complement
    if(cur->parent) cur->parent->children.push_back(e);
    stack.back() = e;
  }

  // CLOSE (s_or exec,exec,save): reconverge and pop. A well-structured region closes at or
  // after the post-dominance reconvergence recorded at open; otherwise it is unbalanced.
  void closeRegion(Address addr, std::vector<Region*>& stack){
    if(stack.size() <= 1) return;                 // never pop the root
    Region* cur = stack.back();
    cur->balanced = (cur->reconverge == 0) || (addr >= cur->reconverge);
    stack.pop_back();
  }

  // NARROW (s_and_b64 exec,exec,cond): a `&&` refinement — AND the extra condition into the
  // current region's condition (short-circuit lowering of `if (A && B)`).
  void narrowRegion(int i, std::vector<Region*>& stack){
    Region* cur = stack.back();
    if(cur == root_) return;
    const std::string c = recoverCond(i);
    cur->cond = cur->cond.empty() ? c : cur->cond + " && " + c;
  }
};
} // namespace CFR

// Print the recovered region tree as indented pseudocode.
static void dump(const CFR::Region *r, int depth){
  const std::string pad(2 * depth, ' ');
  switch(r->kind){
    case CFR::Kind::Root:
      printf("function\n");
      break;
    case CFR::Kind::Else:
      printf("%selse [%s]   // @0x%lx %s\n", pad.c_str(), r->effCond.c_str(),
             (unsigned long)r->header, r->balanced ? "" : "[REROUTE/irregular]");
      break;
    default:  // If (and, later, Loop)
      printf("%sif %s {   // @0x%lx  reconverge@0x%lx %s\n", pad.c_str(), r->effCond.c_str(),
             (unsigned long)r->header, (unsigned long)r->reconverge,
             r->balanced ? "" : "[UNBALANCED]");
      break;
  }
  for(Address a : r->body)
    printf("%s  body @0x%lx\n", pad.c_str(), (unsigned long)a);
  for(CFR::Region *child : r->children)
    dump(child, depth + 1);
}

int main(int argc, char **argv){
  if(argc < 2){
    fprintf(stderr, "usage: %s <code-object> [func-name-substr]\n", argv[0]);
    return 2;
  }
  char *coPath = argv[1];
  const std::string filter = argc >= 3 ? argv[2] : "";

  // Parse the code object into a CFG: SymtabCodeSource reads the ELF, CodeObject builds the
  // functions/blocks/edges, parse() runs ParseAPI. (Leaked deliberately — the process exits.)
  SymtabCodeSource *src = new SymtabCodeSource(coPath);
  CodeObject *co = new CodeObject(src);
  co->parse();

  for(Function *f : co->funcs()){
    if(!filter.empty() && f->name().find(filter) == std::string::npos) continue;

    // Recover the region tree, then print only functions that actually contain divergence —
    // i.e. where at least one region was found (root has children). Straight-line and
    // pure-predication functions leave the root empty. (This replaces a redundant pre-scan
    // that decoded every instruction just to look for one OPEN before decoding them again.)
    CFR::StructureAnalysis sa(f, src->getSymtabObject());
    std::vector<Loop*> loops; f->getLoops(loops);
    if(sa.root()->children.empty() && loops.empty()) continue;

    printf("\n========== %s ==========\n", f->name().c_str());
    if(!sa.root()->children.empty()) dump(sa.root(), 0);
    sa.recoverScalarLoops();
    sa.recoverExecLoops();
  }
  return 0;
}
