# Top-level orchestration for the PoC-owned components. The patched Dyninst lives in the
# dyninst/ submodule and is built/installed separately (it's large) — see README.
#
#   make build       # device runtime + user lib + host (preload/launcher) + mutatees
#   make mutators    # build the Dyninst mutators (needs DYNINST_PREFIX installed)
#   make clean
#
# Toolchain paths + target contract (also included by every sub-Makefile).
# Override DYNINST_PREFIX / ROCM here or on the command line.
include config.mk

.PHONY: build runtime userlib host mutatees mutators clean clean-experiments

build: runtime userlib host mutatees

runtime:                       ## device half of the hostcall runtime
	$(MAKE) -C runtime/device
host:                          ## LD_PRELOAD product + launcher harness
	$(MAKE) -C runtime/host
userlib:                       ## example user instrumentation (+ runtime), one code object
	$(MAKE) -C instrumentation/user_lib
mutatees:                      ## test HIP applications
	$(MAKE) -C mutatees/vectoradd

mutators:                      ## Dyninst mutators (standalone CMake against installed Dyninst)
	cmake -S instrumentation/mutators -B instrumentation/mutators/build \
	      -DDyninst_DIR=$(DYNINST_PREFIX)/lib64/cmake/Dyninst
	cmake --build instrumentation/mutators/build -j

clean:
	-$(MAKE) -C runtime/device clean
	-$(MAKE) -C runtime/host clean
	-$(MAKE) -C instrumentation/user_lib clean
	-$(MAKE) -C mutatees/vectoradd clean
	-rm -rf instrumentation/mutators/build experiments/runs

clean-experiments:             ## remove ONLY experiment outputs (keeps built libs/mutators/EXEs)
	-rm -rf experiments/runs
	-rm -f mutatees/vectoradd/*.co mutatees/vectoradd/*.bundle \
	       mutatees/vectoradd/*.0.hipv4* mutatees/vectoradd/*.0.host-*
