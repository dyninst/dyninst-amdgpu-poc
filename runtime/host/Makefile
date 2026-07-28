# Host half of the hostcall runtime: the two loader-injection front-ends (preload.so via
# LD_PRELOAD, audit.so via LD_AUDIT) and the standalone diagnostic harness (launcher). All
# share hostcall_service.cpp; the two injectors also share hostcall_hooks.h.
include ../../config.mk                    # toolchain paths + target contract
CXXFLAGS  = -O2 -g -std=c++17 -I$(ROCM)/include
LDFLAGS   = -L$(ROCM)/lib -lhsa-runtime64 -lpthread
HOOKS     = hostcall_hooks.h hostcall_service.h co_stride.h ../hostcalls.h

all: preload.so audit.so launcher launcher_nd

preload.so: preload.cpp hostcall_service.cpp $(HOOKS)
	$(CXX) -shared -fPIC $(CXXFLAGS) preload.cpp hostcall_service.cpp -o $@ $(LDFLAGS) -ldl

# audit.so must NOT link libhsa/libamdhip: an LD_AUDIT module lives in its own link-map
# namespace, so linking them would load a SECOND HSA runtime. It resolves the app's real
# functions at run time via dlsym against the app's own libraries (see audit.cpp).
audit.so: audit.cpp hostcall_service.cpp $(HOOKS)
	$(CXX) -shared -fPIC $(CXXFLAGS) audit.cpp hostcall_service.cpp -o $@ -lpthread -ldl

launcher: launcher.cpp hostcall_service.cpp hostcall_service.h ../hostcalls.h
	$(CXX) $(CXXFLAGS) launcher.cpp hostcall_service.cpp -o $@ $(LDFLAGS) -lamdhip64

# N-D (2D/3D) launcher for debugging the per-wave wid; raw HSA, no HIP dependency.
launcher_nd: launcher_nd.cpp hostcall_service.cpp hostcall_service.h ../hostcalls.h co_stride.h
	$(CXX) $(CXXFLAGS) launcher_nd.cpp hostcall_service.cpp -o $@ $(LDFLAGS)

clean:
	rm -f preload.so audit.so launcher launcher_nd
