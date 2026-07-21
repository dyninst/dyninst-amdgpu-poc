# config.mk — single source of toolchain paths + the target contract for ALL Makefiles
# in this repo. Each build dir is two levels deep, so include it as:
#     include ../../config.mk
# The shell counterpart used by scripts/ and experiments/ is scripts/env.sh — keep the
# two in sync. Everything is ?= so it can be overridden from the env or the make cmdline.

# Absolute path to this file's directory (the repo root). Robust regardless of which
# sub-Makefile includes us: each references config.mk by its own relative path, and
# $(abspath ...) resolves that against the includer's CWD. Evaluated immediately (:=)
# while config.mk is the last entry in MAKEFILE_LIST.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

ROCM           ?= /opt/rocm-7.0.2
# Patched Dyninst install prefix — repo-relative by default (built from the dyninst/
# submodule into <repo>/build/dyninst), resolved to an absolute path so it works in
# LD_LIBRARY_PATH / DYNINSTAPI_RT_LIB. Override to reuse an existing install.
DYNINST_PREFIX ?= $(REPO_ROOT)/build/dyninst

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
