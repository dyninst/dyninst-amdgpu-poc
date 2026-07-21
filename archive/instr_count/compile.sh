#!/bin/bash
# Build the count_inst instrumentation library into a gfx908 code object and add
# the .dyninst.* STT_OBJECT alias the cross-object relocation resolves against.
set -e
cd "$(dirname "$0")"

ARCH=gfx908
ROCM=/opt/rocm-7.0.2
export PATH="$ROCM/bin:$ROCM/llvm/bin:$PATH"

# 1. Generate a HIP fat-binary bundle (code object only, no host).
hipcc --genco --offload-arch=$ARCH count_inst.cpp -o count_inst.bundle

# 2. Extract the raw gfx908 code object from the bundle.
clang-offload-bundler --unbundle --input=./count_inst.bundle --type=o \
  --targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa--$ARCH \
  --outputs=/dev/null,count_inst.raw.hsaco

# 3. Add .dyninst.<func> STT_OBJECT aliases so dyninst's inserted calls resolve.
python3 ../add_object_aliases.py count_inst.raw.hsaco count_inst_gfx908.hsaco

echo "built count_inst_gfx908.hsaco"
