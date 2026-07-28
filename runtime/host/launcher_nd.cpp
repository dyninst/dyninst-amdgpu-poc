// launcher_nd — raw-HSA, rocgdb-attachable launcher for 2D/3D instrumented kernels.
//
// Same load+link+service flow as launcher.cpp, but dispatches an N-D grid and fills the
// COV5 hidden implicit args (block_count_{x,y,z}, group_size_{x,y,z}) at the kernel's
// metadata offsets, so blockDim/gridDim are correct for a 2D/3D kernel. Purpose: reproduce
// and debug the per-wave global-wavefront-id under gdb (the preload path hides the dispatch).
//
// Config (env):
//   BX,BY,BZ   workgroup (block) dims        (default 8,8,4)
//   GX,GY,GZ   grid dims IN BLOCKS           (default 2,2,2)
//   (The COV5 hidden-arg offsets block_count_x / group_size_x are read from the co's OWN
//    .note metadata — self-describing, like __dyninst_pw_stride — not from any env var.)
//   NDIM       dispatch dimensionality 2 or 3 (default 3)
// Args: <mutatee.inst.co> <lib.aliased.elf> <kd_name>
//   e.g. launcher_nd vectoradd3d.inst.synced.co combined.aliased.elf _Z11vectoradd3dPfPKfS1_iii.kd
//
// Run with ROCR_VISIBLE_DEVICES=1 (gfx908 MI100).

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <atomic>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

#include "../hostcalls.h"
#include "hostcall_service.h"
#include "co_stride.h"

#define HSA_CHECK(s) do {                                                      \
    hsa_status_t _s = (s);                                                     \
    if (_s != HSA_STATUS_SUCCESS) {                                            \
        const char* _m = nullptr; hsa_status_string(_s, &_m);                 \
        fprintf(stderr, "HSA error: %s at %s:%d\n", _m, __FILE__, __LINE__);   \
        exit(1);                                                               \
    }                                                                          \
} while (0)

struct AgentSearch { hsa_agent_t agent; bool found; };
static hsa_status_t find_gpu(hsa_agent_t a, void* d) {
    hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
    char n[64] = {}; hsa_agent_get_info(a, HSA_AGENT_INFO_NAME, n);
    if (strstr(n, "gfx908")) { auto* s=(AgentSearch*)d; s->agent=a; s->found=true; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t find_cpu(hsa_agent_t a, void* d) {
    hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_CPU) { auto* s=(AgentSearch*)d; s->agent=a; s->found=true; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}
struct PoolSearch { hsa_amd_memory_pool_t pool; bool found; };
static hsa_status_t find_fine_grained(hsa_amd_memory_pool_t pool, void* d) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
        auto* s=(PoolSearch*)d; s->pool=pool; s->found=true; return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}
static std::atomic<bool> g_run{true};

static uint32_t envu(const char* k, uint32_t dflt) { const char* e=getenv(k); return e?(uint32_t)strtoul(e,nullptr,0):dflt; }

int main(int argc, char** argv) {
    const char* mutatee = (argc > 1) ? argv[1] : "vectoradd3d.inst.synced.co";
    const char* instlib = (argc > 2) ? argv[2] : "combined.aliased.elf";
    const char* kd_name = (argc > 3) ? argv[3] : "_Z11vectoradd3dPfPKfS1_iii.kd";

    const uint32_t BX=envu("BX",8), BY=envu("BY",8), BZ=envu("BZ",4);      // block dims
    const uint32_t GX=envu("GX",2), GY=envu("GY",2), GZ=envu("GZ",2);      // grid dims (blocks)
    const uint32_t NDIM=envu("NDIM",3);
    // Self-describing: read the COV5 hidden-arg offsets from the instrumented co's OWN
    // metadata note (like __dyninst_pw_stride below), not a hardcoded per-kernel constant.
    const uint32_t OFF_BC = co_read_kernarg_offset(mutatee, "hidden_block_count_x", 0xffffffffu);
    const uint32_t OFF_GS = co_read_kernarg_offset(mutatee, "hidden_group_size_x",  0xffffffffu);
    if (OFF_BC==0xffffffffu || OFF_GS==0xffffffffu) {
        fprintf(stderr, "[host] could not read hidden-arg offsets from %s's .note\n", mutatee);
        return 1;
    }
    printf("[host] hidden-arg offsets from co: block_count@%u group_size@%u\n", OFF_BC, OFF_GS);
    const uint32_t WX=BX*GX, WY=BY*GY, WZ=BZ*GZ;                            // grid sizes (threads)
    const uint32_t N = WX*WY*WZ;
    const uint32_t threadsPerBlk = BX*BY*BZ;
    const uint32_t wpb = (threadsPerBlk + 63)/64;
    const uint32_t n_blocks = GX*GY*GZ;
    const uint32_t n_waves = n_blocks * wpb;

    HSA_CHECK(hsa_init());
    AgentSearch gpu={}, cpu={};
    hsa_iterate_agents(find_gpu, &gpu); hsa_iterate_agents(find_cpu, &cpu);
    if (!gpu.found || !cpu.found) { fprintf(stderr, "no gfx908 GPU / CPU agent\n"); return 1; }
    PoolSearch fg={};
    hsa_amd_agent_iterate_memory_pools(cpu.agent, find_fine_grained, &fg);
    if (!fg.found) { fprintf(stderr, "no fine-grained pool\n"); return 1; }
    printf("[host] gfx908 + fine-grained pool; dims block(%u,%u,%u) grid(%u,%u,%u) => "
           "N=%u, %u blocks x %u waves/block = %u waves\n",
           BX,BY,BZ,GX,GY,GZ,N,n_blocks,wpb,n_waves);

    HostcallMailbox* mbox = nullptr;
    HSA_CHECK(hsa_amd_memory_pool_allocate(fg.pool, sizeof(HostcallMailbox), 0, (void**)&mbox));
    HSA_CHECK(hsa_amd_agents_allow_access(1, &gpu.agent, nullptr, mbox));
    memset(mbox, 0, sizeof(*mbox));
    for (uint32_t i = 0; i < HC_NSLOTS; i++) mbox->slots[i].turn = i;

    hsa_executable_t exe;
    HSA_CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, "", &exe));
    HSA_CHECK(hsa_executable_agent_global_variable_define(exe, gpu.agent, "mailbox", mbox));
    auto load = [&](const char* path) {
        int fd = open(path, O_RDONLY); if (fd < 0) { perror(path); exit(1); }
        hsa_code_object_reader_t r;
        HSA_CHECK(hsa_code_object_reader_create_from_file(fd, &r));
        HSA_CHECK(hsa_executable_load_agent_code_object(exe, gpu.agent, r, "", nullptr));
        close(fd); printf("[host] loaded %s\n", path);
    };
    load(instlib); load(mutatee);
    HSA_CHECK(hsa_executable_freeze(exe, ""));
    printf("[host] frozen (cross-object relocs resolved)\n");

    hsa_executable_symbol_t ksym;
    HSA_CHECK(hsa_executable_get_symbol_by_name(exe, kd_name, &gpu.agent, &ksym));
    uint64_t kobj; uint32_t ksize, psize, gsize;
    HSA_CHECK(hsa_executable_symbol_get_info(ksym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kobj));
    HSA_CHECK(hsa_executable_symbol_get_info(ksym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &ksize));
    HSA_CHECK(hsa_executable_symbol_get_info(ksym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &psize));
    HSA_CHECK(hsa_executable_symbol_get_info(ksym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &gsize));
    printf("[host] kernel: kobj=%#lx ksize=%u psize=%u gsize=%u\n", kobj, ksize, psize, gsize);

    // Data buffers (fine-grained). vectoradd{2,3}d(C, A, B, W, H[, D]).
    float *A, *B, *C;
    HSA_CHECK(hsa_amd_memory_pool_allocate(fg.pool, N*sizeof(float), 0, (void**)&A));
    HSA_CHECK(hsa_amd_memory_pool_allocate(fg.pool, N*sizeof(float), 0, (void**)&B));
    HSA_CHECK(hsa_amd_memory_pool_allocate(fg.pool, N*sizeof(float), 0, (void**)&C));
    hsa_agent_t agents[1] = { gpu.agent };
    HSA_CHECK(hsa_amd_agents_allow_access(1, agents, nullptr, A));
    HSA_CHECK(hsa_amd_agents_allow_access(1, agents, nullptr, B));
    HSA_CHECK(hsa_amd_agents_allow_access(1, agents, nullptr, C));
    for (uint32_t i=0;i<N;i++){ A[i]=(float)i; B[i]=(float)(2*i); C[i]=0; }

    const uint32_t STRIDE = co_read_symbol_u32(mutatee, "__dyninst_pw_stride", 4096);
    printf("[host] per-wave STRIDE = %u B\n", STRIDE);
    void* instbuf = nullptr;
    HSA_CHECK(hsa_amd_memory_pool_allocate(fg.pool, (size_t)n_waves*STRIDE, 0, &instbuf));
    HSA_CHECK(hsa_amd_agents_allow_access(1, agents, nullptr, instbuf));
    memset(instbuf, 0, (size_t)n_waves*STRIDE);

    // Kernarg: explicit C,A,B,W,H[,D] + hidden block_count/group_size + appended instbuf ptr.
    void* ka = nullptr; uint32_t ka_sz = ksize < 64 ? 64 : ksize;
    HSA_CHECK(hsa_amd_memory_pool_allocate(fg.pool, ka_sz, 0, &ka));
    HSA_CHECK(hsa_amd_agents_allow_access(1, agents, nullptr, ka));
    memset(ka, 0, ka_sz);
    char* k = (char*)ka;
    *(float**)(k+0)=C; *(const float**)(k+8)=A; *(const float**)(k+16)=B;
    *(int*)(k+24)=(int)WX; *(int*)(k+28)=(int)WY;          // W, H
    if (NDIM==3) *(int*)(k+32)=(int)WZ;                    // D
    *(uint32_t*)(k+OFF_BC+0)=GX; *(uint32_t*)(k+OFF_BC+4)=GY; *(uint32_t*)(k+OFF_BC+8)=GZ; // block_count
    *(uint16_t*)(k+OFF_GS+0)=(uint16_t)BX; *(uint16_t*)(k+OFF_GS+2)=(uint16_t)BY; *(uint16_t*)(k+OFF_GS+4)=(uint16_t)BZ; // group_size
    if (ksize >= 8) *(void**)(k+(ksize-8)) = instbuf;      // dyninst per-wave buffer ptr

    hsa_queue_t* queue = nullptr;
    HSA_CHECK(hsa_queue_create(gpu.agent, 256, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue));
    hsa_signal_t done; HSA_CHECK(hsa_signal_create(1, 0, nullptr, &done));
    std::thread svc(hostcall_service_loop, mbox, std::ref(g_run), "host");
    printf("[host] service thread started; dispatching (NDIM=%u)\n", NDIM);

    uint64_t idx = hsa_queue_load_write_index_relaxed(queue);
    const uint32_t mask = queue->size - 1;
    hsa_kernel_dispatch_packet_t* slot = &((hsa_kernel_dispatch_packet_t*)queue->base_address)[idx & mask];
    memset((void*)((uintptr_t)slot + 4), 0, sizeof(*slot) - 4);
    slot->setup = NDIM << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    slot->workgroup_size_x=(uint16_t)BX; slot->workgroup_size_y=(uint16_t)BY; slot->workgroup_size_z=(uint16_t)BZ;
    slot->grid_size_x=WX; slot->grid_size_y=WY; slot->grid_size_z=WZ;
    slot->private_segment_size=psize; slot->group_segment_size=gsize;
    slot->kernel_object=kobj; slot->kernarg_address=ka; slot->completion_signal=done;
    uint16_t header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    __atomic_store_n(&slot->header, header, __ATOMIC_RELEASE);
    hsa_queue_store_write_index_relaxed(queue, idx + 1);
    hsa_signal_store_screlease(queue->doorbell_signal, idx);

    hsa_signal_value_t v = hsa_signal_wait_scacquire(done, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)10e9, HSA_WAIT_STATE_BLOCKED);
    printf("[host] kernel completion = %ld\n", (long)v);
    g_run.store(false, std::memory_order_release); svc.join();

    // Per-wave slice dump: a correct wid bijection writes ALL n_waves slices; a collapsed
    // wave-in-block (the bug) leaves only n_blocks distinct slices non-zero.
    int nonzero = 0;
    for (uint32_t w = 0; w < n_waves; w++) {
        const uint32_t* pw = (const uint32_t*)((char*)instbuf + (size_t)w*STRIDE);
        if (pw[0]||pw[1]||pw[2]||pw[3]) nonzero++;
    }
    printf("[host] per-wave slices written: %d / %u  (expect all %u if wid is a bijection)\n",
           nonzero, n_waves, n_waves);

    int errors = 0;
    for (uint32_t i=0;i<N;i++) if (C[i] != A[i]+B[i]) errors++;
    printf(errors ? "[host] vectoradd FAILED (%d errors)\n" : "[host] vectoradd PASSED (%u elems)\n",
           errors ? errors : N);

    hsa_signal_destroy(done); hsa_queue_destroy(queue); hsa_executable_destroy(exe); hsa_shut_down();
    return errors ? 1 : 0;
}
