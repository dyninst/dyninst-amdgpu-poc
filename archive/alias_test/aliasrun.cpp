// Minimal HSA runner for the flat-vs-buffer scratch aliasing test.
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#define HSA_CHECK(s) do{ hsa_status_t _s=(s); if(_s!=HSA_STATUS_SUCCESS){ const char*_m=nullptr; \
  hsa_status_string(_s,&_m); fprintf(stderr,"HSA err %s @%d\n",_m,__LINE__); exit(1);} }while(0)
struct AS{ hsa_agent_t a; bool f; };
static hsa_status_t find_gpu(hsa_agent_t a,void*d){ hsa_device_type_t t; hsa_agent_get_info(a,HSA_AGENT_INFO_DEVICE,&t);
  if(t!=HSA_DEVICE_TYPE_GPU)return HSA_STATUS_SUCCESS; char n[64]={}; hsa_agent_get_info(a,HSA_AGENT_INFO_NAME,n);
  if(strstr(n,"gfx908")){auto*s=(AS*)d;s->a=a;s->f=true;return HSA_STATUS_INFO_BREAK;} return HSA_STATUS_SUCCESS; }
struct PS{ hsa_amd_memory_pool_t p; bool f; };
static hsa_status_t find_fg(hsa_amd_memory_pool_t p,void*d){ hsa_amd_segment_t seg;
  hsa_amd_memory_pool_get_info(p,HSA_AMD_MEMORY_POOL_INFO_SEGMENT,&seg); if(seg!=HSA_AMD_SEGMENT_GLOBAL)return HSA_STATUS_SUCCESS;
  uint32_t fl; hsa_amd_memory_pool_get_info(p,HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,&fl);
  if(fl&HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED){auto*s=(PS*)d;s->p=p;s->f=true;return HSA_STATUS_INFO_BREAK;} return HSA_STATUS_SUCCESS; }
int main(int argc,char**argv){
  const char* co = argc>1?argv[1]:"alias.co";
  int NT = argc>2?atoi(argv[2]):64;                 // threads (=grid=workgroup); 64=1 wave,128=2 waves
  HSA_CHECK(hsa_init());
  AS gpu={},cpu={}; hsa_iterate_agents(find_gpu,&gpu);
  { hsa_iterate_agents([](hsa_agent_t a,void*d)->hsa_status_t{ hsa_device_type_t t; hsa_agent_get_info(a,HSA_AGENT_INFO_DEVICE,&t);
      if(t==HSA_DEVICE_TYPE_CPU){auto*s=(AS*)d;s->a=a;s->f=true;return HSA_STATUS_INFO_BREAK;} return HSA_STATUS_SUCCESS;},&cpu); }
  if(!gpu.f||!cpu.f){fprintf(stderr,"no agent\n");return 1;}
  PS fg={}; hsa_amd_agent_iterate_memory_pools(cpu.a,find_fg,&fg); if(!fg.f){fprintf(stderr,"no fg pool\n");return 1;}
  hsa_executable_t exe;
  HSA_CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL,HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,"",&exe));
  int fd=open(co,O_RDONLY); if(fd<0){perror(co);return 1;}
  hsa_code_object_reader_t r; HSA_CHECK(hsa_code_object_reader_create_from_file(fd,&r));
  HSA_CHECK(hsa_executable_load_agent_code_object(exe,gpu.a,r,"",nullptr)); close(fd);
  HSA_CHECK(hsa_executable_freeze(exe,""));
  hsa_executable_symbol_t ks;
  HSA_CHECK(hsa_executable_get_symbol_by_name(exe,"aliastest.kd",&gpu.a,&ks));
  uint64_t kobj; uint32_t ksz,psz,gsz;
  HSA_CHECK(hsa_executable_symbol_get_info(ks,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,&kobj));
  HSA_CHECK(hsa_executable_symbol_get_info(ks,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,&ksz));
  HSA_CHECK(hsa_executable_symbol_get_info(ks,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,&psz));
  HSA_CHECK(hsa_executable_symbol_get_info(ks,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,&gsz));
  printf("kobj=%#lx ksize=%u psize=%u gsize=%u  threads=%d waves=%d\n",kobj,ksz,psz,gsz,NT,(NT+63)/64);
  hsa_agent_t ag[1]={gpu.a};
  uint32_t* out=nullptr; size_t osz=(size_t)NT*5*sizeof(uint32_t);
  HSA_CHECK(hsa_amd_memory_pool_allocate(fg.p,osz,0,(void**)&out));
  HSA_CHECK(hsa_amd_agents_allow_access(1,ag,nullptr,out)); memset(out,0,osz);
  void* ka=nullptr; uint32_t kasz=ksz<64?64:ksz;
  HSA_CHECK(hsa_amd_memory_pool_allocate(fg.p,kasz,0,&ka));
  HSA_CHECK(hsa_amd_agents_allow_access(1,ag,nullptr,ka)); memset(ka,0,kasz);
  *(uint32_t**)ka = out;
  hsa_queue_t* q=nullptr; HSA_CHECK(hsa_queue_create(gpu.a,256,HSA_QUEUE_TYPE_SINGLE,nullptr,nullptr,UINT32_MAX,UINT32_MAX,&q));
  hsa_signal_t done; HSA_CHECK(hsa_signal_create(1,0,nullptr,&done));
  uint64_t idx=hsa_queue_load_write_index_relaxed(q); const uint32_t mask=q->size-1;
  auto* slot=&((hsa_kernel_dispatch_packet_t*)q->base_address)[idx&mask];
  memset((void*)((uintptr_t)slot+4),0,sizeof(*slot)-4);
  slot->setup=1<<HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  slot->workgroup_size_x=NT; slot->workgroup_size_y=1; slot->workgroup_size_z=1;
  slot->grid_size_x=NT; slot->grid_size_y=1; slot->grid_size_z=1;
  slot->private_segment_size=psz; slot->group_segment_size=gsz;
  slot->kernel_object=kobj; slot->kernarg_address=ka; slot->completion_signal=done;
  uint16_t h=(HSA_PACKET_TYPE_KERNEL_DISPATCH<<HSA_PACKET_HEADER_TYPE)|(1<<HSA_PACKET_HEADER_BARRIER)|
    (HSA_FENCE_SCOPE_SYSTEM<<HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE)|(HSA_FENCE_SCOPE_SYSTEM<<HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  __atomic_store_n(&slot->header,h,__ATOMIC_RELEASE);
  hsa_queue_store_write_index_relaxed(q,idx+1); hsa_signal_store_screlease(q->doorbell_signal,idx);
  hsa_signal_wait_scacquire(done,HSA_SIGNAL_CONDITION_LT,1,(uint64_t)5e9,HSA_WAIT_STATE_BLOCKED);
  // print a few representative lanes
  printf("lane : sentA sentB   L0(soff0) L4(soff4) L256(soff256)\n");
  int lanes[]={0,1,2,63,64,65,127};
  for(int li=0; li<7; li++){ int L=lanes[li]; if(L>=NT)continue; uint32_t*o=out+L*5;
    printf("%4d : %5x %5x   %8x %8x %8x\n",L,o[0],o[1],o[2],o[3],o[4]); }
  return 0;
}
