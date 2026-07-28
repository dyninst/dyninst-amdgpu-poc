# Test HIP applications (mutatees). Built with the SAME feature set as the runtime/
# user lib (gfx908:sramecc+:xnack- + COV6 => ABI ver 4, flags 0xE30) so cross-object
# linking works. Code objects are extracted by scripts/build_inst_pipeline.sh.
include ../../config.mk                    # toolchain paths + target contract
HIPFLAGS = $(HIP_BASEFLAGS)

EXES = vectoradd vectoradd_small vectoradd_mw vectoradd2d vectoradd3d twokernels twoco bbdemo bbmw

all: $(EXES)

# single-kernel vectoradd (N=1<<20) + small/multi-wave/2D/3D variants; bbdemo/bbmw = BB-counter demos
vectoradd vectoradd_small vectoradd_mw vectoradd2d vectoradd3d twokernels bbdemo bbmw: %: %.cpp
	$(HIPCC) $(HIPFLAGS) $< -o $@

# multi-code-object: kernel kB in a HIP shared library, kernel kA in the exe
libkb.so: kb_lib.cpp
	$(HIPCC) $(HIPFLAGS) -fPIC -shared $< -o $@
twoco: twoco.cpp libkb.so
	$(HIPCC) $(HIPFLAGS) $< -L. -lkb -Wl,-rpath,'$$ORIGIN' -o $@

clean:
	rm -f $(EXES) libkb.so *.co *.bundle *.0.hipv4* *.0.host-*
