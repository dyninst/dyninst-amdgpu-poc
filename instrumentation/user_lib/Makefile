# Example user instrumentation compiled TOGETHER with the hostcall runtime into ONE
# code object (combined.aliased.elf) — the artifact dyninst links a mutatee against.
include ../../config.mk                    # toolchain paths + target contract
HIPFLAGS = $(HIP_BASEFLAGS) $(HIP_LINKIGNORE)
# hostcall runtime (hostcalls.h + device/hostcall_lib.cpp)
RT       = ../../runtime
SRCS     = user_probe.cpp $(RT)/device/hostcall_lib.cpp
ALIASES  = ../../tools/add_object_aliases.py

all: combined.aliased.elf

# -fgpu-rdc device-links both TUs into one code object (user_probe's calls into the
# runtime resolve internally). Compiling both as ONE TU for the .s makes non-leaf
# register-usage expressions reference concrete callee symbols (see add_object_aliases).
combined.hsaco: $(SRCS) $(RT)/hostcalls.h
	$(HIPCC) --genco -fgpu-rdc $(HIPFLAGS) -I$(RT) $(SRCS) -o $@
combined.s: $(SRCS) $(RT)/hostcalls.h
	cat user_probe.cpp $(RT)/device/hostcall_lib.cpp > combined_tu.cpp
	$(HIPCC) -S --cuda-device-only $(HIPFLAGS) -I$(RT) combined_tu.cpp -o $@
combined.elf: combined.hsaco
	$(BUNDLER) --type=o --targets=hipv4-amdgcn-amd-amdhsa--$(ARCH) \
	  --input=combined.hsaco --output=$@ --unbundle
combined.aliased.elf: combined.elf combined.s $(ALIASES)
	python3 $(ALIASES) combined.elf $@ \
	  --asm combined.s --tables symtab --obj2yaml $(OBJ2YAML) --yaml2obj $(YAML2OBJ)

clean:
	rm -f combined.hsaco combined.s combined_tu.cpp combined.elf combined.aliased.elf
