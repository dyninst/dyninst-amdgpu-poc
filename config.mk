# config.mk — single source of toolchain paths + the target contract for ALL Makefiles
# in this repo. Each build dir is two levels deep, so include it as:
#     include ../../config.mk
# The shell counterpart used by scripts/ and experiments/ is scripts/env.sh — keep the
# two in sync. Everything is ?= so it can be overridden from the env or the make cmdline.

ROCM           ?= /opt/rocm-7.0.2
DYNINST_PREFIX ?= $(HOME)/bin/dynamd

# ---- toolchain ----
HIPCC    ?= $(ROCM)/bin/hipcc
BUNDLER  ?= $(ROCM)/lib/llvm/bin/clang-offload-bundler
OBJDUMP  ?= $(ROCM)/lib/llvm/bin/llvm-objdump
OBJ2YAML ?= /usr/bin/obj2yaml
YAML2OBJ ?= /usr/bin/yaml2obj
CXX      ?= g++

# ---- target contract ----
# Full gfx908 target-id + COV6 => ABI ver 4, feature flags 0xE30. EVERY co-loaded object
# (hostcall runtime, user lib, mutatees) MUST agree on arch + COV + features, or the HSA
# executable freeze rejects the cross-object link.
ARCH ?= gfx908:sramecc+:xnack-
COV  ?= 6

# Base device-compile flags. The runtime + user lib carry undefined externs (`mailbox`,
# `g_pw_base`) resolved at load, hence ignore-all; mutatees are complete apps and add
# only HIP_BASEFLAGS.
HIP_BASEFLAGS  ?= --offload-arch=$(ARCH) -mcode-object-version=$(COV)
HIP_LINKIGNORE ?= -Xoffload-linker --unresolved-symbols=ignore-all
