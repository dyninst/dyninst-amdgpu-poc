# PoC-owned Dyninst mutators. Standalone CMake project built against an installed
# (patched) Dyninst — the one from the dyninst/ submodule:
#   cmake -S . -B build -DDyninst_DIR=<prefix>/lib64/cmake/Dyninst
#   cmake --build build
cmake_minimum_required(VERSION 3.14)
project(amdgpu_poc_mutators LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Dyninst REQUIRED COMPONENTS dyninstAPI)

# Each mutator is a standalone binary that rewrites a GPU code object, inserting calls
# into the hostcall runtime / user library.
set(MUTATORS
  test_amdgpu_instrument      # full capability demonstrator
  minimal_instrument          # canonical minimal example (record EXEC at a few sites)
  preload_perwave_instrument  # per-wave multi-file (pw_open/pw_flush via kernarg PerWaveBuf)
  multikernel_instrument      # instrument several kernels in one code object
  capture_return_instrument   # capture a call's ABI return into a per-wave variable
  bb_count_instrument         # basic-block execution counter (getpc-free kernels), file flush
  bb_count_multivar_instrument # same counter, one per-wave var PER block (multi-var style)
  real_write_instrument       # pass-by-address vs by-value fwrite egress eval
  multivar_instrument         # multiple per-wave variables at distinct arena offsets
  getpc_reloc_instrument      # validate getpc-call relocation (getpc-idiom kernel)
)
foreach(m ${MUTATORS})
  add_executable(${m} ${m}.C)
  target_link_libraries(${m} PRIVATE Dyninst::dyninstAPI)
endforeach()
