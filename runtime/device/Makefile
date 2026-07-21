# Device half of the hostcall runtime -> hostcall_lib.aliased.elf
# (raw code object + .dyninst.* OBJECT aliases + per-function register-usage symbols).
include ../../config.mk                    # toolchain paths + target contract
HIPFLAGS = $(HIP_BASEFLAGS) -I.. $(HIP_LINKIGNORE)
ALIASES  = ../../tools/add_object_aliases.py

all: hostcall_lib.aliased.elf

hostcall_lib.hsaco: hostcall_lib.cpp ../hostcalls.h
	$(HIPCC) --genco $(HIPFLAGS) hostcall_lib.cpp -o $@
hostcall_lib.s: hostcall_lib.cpp ../hostcalls.h
	$(HIPCC) -S --cuda-device-only $(HIPFLAGS) hostcall_lib.cpp -o $@
hostcall_lib.elf: hostcall_lib.hsaco
	$(BUNDLER) --type=o --targets=hipv4-amdgcn-amd-amdhsa--$(ARCH) \
	  --input=hostcall_lib.hsaco --output=$@ --unbundle
hostcall_lib.aliased.elf: hostcall_lib.elf hostcall_lib.s $(ALIASES)
	python3 $(ALIASES) hostcall_lib.elf $@ \
	  --asm hostcall_lib.s --tables symtab --obj2yaml $(OBJ2YAML) --yaml2obj $(YAML2OBJ)

clean:
	rm -f hostcall_lib.hsaco hostcall_lib.s hostcall_lib.elf hostcall_lib.aliased.elf
