#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#define CK(s) do{hsa_status_t _s=(s); if(_s){const char*m; hsa_status_string(_s,&m); fprintf(stderr,"HSA %s @%d\n",m,__LINE__); exit(1);}}while(0)
struct A{hsa_agent_t a;bool f;};
static hsa_status_t fg_(hsa_agent_t a,void*d){hsa_device_type_t t;hsa_agent_get_info(a,HSA_AGENT_INFO_DEVICE,&t);if(t!=HSA_DEVICE_TYPE_GPU)return HSA_STATUS_SUCCESS;char n[64]={};hsa_agent_get_info(a,HSA_AGENT_INFO_NAME,n);if(strstr(n,"gfx908")){((A*)d)->a=a;((A*)d)->f=1;return HSA_STATUS_INFO_BREAK;}return HSA_STATUS_SUCCESS;}
struct P{hsa_amd_memory_pool_t p;bool f;};
static hsa_status_t fp_(hsa_amd_memory_pool_t p,void*d){hsa_amd_segment_t s;hsa_amd_memory_pool_get_info(p,HSA_AMD_MEMORY_POOL_INFO_SEGMENT,&s);if(s!=HSA_AMD_SEGMENT_GLOBAL)return HSA_STATUS_SUCCESS;uint32_t fl;hsa_amd_memory_pool_get_info(p,HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,&fl);if(fl&HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED){((P*)d)->p=p;((P*)d)->f=1;return HSA_STATUS_INFO_BREAK;}return HSA_STATUS_SUCCESS;}
int main(int c,char**v){int NT=c>1?atoi(v[1]):128;CK(hsa_init());A g={},cpu={};hsa_iterate_agents(fg_,&g);
 hsa_iterate_agents([](hsa_agent_t a,void*d)->hsa_status_t{hsa_device_type_t t;hsa_agent_get_info(a,HSA_AGENT_INFO_DEVICE,&t);if(t==HSA_DEVICE_TYPE_CPU){((A*)d)->a=a;((A*)d)->f=1;return HSA_STATUS_INFO_BREAK;}return HSA_STATUS_SUCCESS;},&cpu);
 P fg={};hsa_amd_agent_iterate_memory_pools(cpu.a,fp_,&fg);
 hsa_executable_t e;CK(hsa_executable_create_alt(HSA_PROFILE_FULL,HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,"",&e));
 int fd=open("desc.co",O_RDONLY);hsa_code_object_reader_t r;CK(hsa_code_object_reader_create_from_file(fd,&r));
 CK(hsa_executable_load_agent_code_object(e,g.a,r,"",0));close(fd);CK(hsa_executable_freeze(e,""));
 hsa_executable_symbol_t k;CK(hsa_executable_get_symbol_by_name(e,"aliastest.kd",&g.a,&k));
 uint64_t ko;uint32_t ks,ps,gs;CK(hsa_executable_symbol_get_info(k,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,&ko));
 CK(hsa_executable_symbol_get_info(k,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,&ks));
 CK(hsa_executable_symbol_get_info(k,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,&ps));
 CK(hsa_executable_symbol_get_info(k,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,&gs));
 hsa_agent_t ag[1]={g.a};uint32_t*out;size_t osz=(size_t)NT*24;CK(hsa_amd_memory_pool_allocate(fg.p,osz,0,(void**)&out));
 CK(hsa_amd_agents_allow_access(1,ag,0,out));memset(out,0,osz);
 void*ka;CK(hsa_amd_memory_pool_allocate(fg.p,64,0,&ka));CK(hsa_amd_agents_allow_access(1,ag,0,ka));memset(ka,0,64);*(uint32_t**)ka=out;
 hsa_queue_t*q;CK(hsa_queue_create(g.a,256,HSA_QUEUE_TYPE_SINGLE,0,0,UINT32_MAX,UINT32_MAX,&q));
 hsa_signal_t d;CK(hsa_signal_create(1,0,0,&d));uint64_t i=hsa_queue_load_write_index_relaxed(q);uint32_t m=q->size-1;
 auto*s=&((hsa_kernel_dispatch_packet_t*)q->base_address)[i&m];memset((void*)((uintptr_t)s+4),0,sizeof(*s)-4);
 s->setup=1<<HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;s->workgroup_size_x=NT;s->workgroup_size_y=1;s->workgroup_size_z=1;
 s->grid_size_x=NT;s->grid_size_y=1;s->grid_size_z=1;s->private_segment_size=ps;s->group_segment_size=gs;s->kernel_object=ko;s->kernarg_address=ka;s->completion_signal=d;
 uint16_t h=(HSA_PACKET_TYPE_KERNEL_DISPATCH<<HSA_PACKET_HEADER_TYPE)|(1<<HSA_PACKET_HEADER_BARRIER)|(HSA_FENCE_SCOPE_SYSTEM<<HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE)|(HSA_FENCE_SCOPE_SYSTEM<<HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
 __atomic_store_n(&s->header,h,__ATOMIC_RELEASE);hsa_queue_store_write_index_relaxed(q,i+1);hsa_signal_store_screlease(q->doorbell_signal,i);
 hsa_signal_wait_scacquire(d,HSA_SIGNAL_CONDITION_LT,1,(uint64_t)5e9,HSA_WAIT_STATE_BLOCKED);
 printf("lane :  FLAT_lo  FLAT_hi | desc_s0  desc_s1  desc_s2   desc_s3\n");
 int L[]={0,1,64,65}; for(int j=0;j<4;j++){int l=L[j];if(l>=NT)continue;uint32_t*o=out+l*6;
   printf("%4d : %08x %08x | %08x %08x %08x %08x\n",l,o[0],o[1],o[2],o[3],o[4],o[5]);}
 return 0;}
