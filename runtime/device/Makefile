# Device half of the hostcall runtime -> hostcall_lib.aliased.elf
# (raw code object + .dyninst.* OBJECT aliases + per-function register-usage symbols).
HIPCC    = /opt/rocm-7.0.2/bin/hipcc
BUNDLER  = /opt/rocm-7.0.2/lib/llvm/bin/clang-offload-bundler
OBJ2YAML = /usr/bin/obj2yaml
YAML2OBJ = /usr/bin/yaml2obj
ARCH     = gfx908:sramecc+:xnack-        # full target-id + COV6 => ABI ver 4, flags 0xE30
COV      = 6
HIPFLAGS = --offload-arch=$(ARCH) -mcode-object-version=$(COV) -I.. \
           -Xoffload-linker --unresolved-symbols=ignore-all
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
