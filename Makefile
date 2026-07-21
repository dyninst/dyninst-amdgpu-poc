# Top-level orchestration for the PoC-owned components. The patched Dyninst lives in the
# dyninst/ submodule and is built/installed separately (it's large) — see README.
#
#   make build       # device runtime + user lib + host (preload/launcher) + mutatees
#   make mutators    # build the Dyninst mutators (needs DYNINST_PREFIX installed)
#   make clean
#
# Override the Dyninst install prefix if it differs:
DYNINST_PREFIX ?= /home/wuxx1279/bin/dynamd

.PHONY: build runtime userlib host mutatees mutators clean

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
