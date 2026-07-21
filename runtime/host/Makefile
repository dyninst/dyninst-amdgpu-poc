# Host half of the hostcall runtime: the LD_PRELOAD product (preload.so) and the
# standalone diagnostic harness (launcher). Both share hostcall_service.cpp.
include ../../config.mk                    # toolchain paths + target contract
CXXFLAGS  = -O2 -g -std=c++17 -I$(ROCM)/include
LDFLAGS   = -L$(ROCM)/lib -lhsa-runtime64 -lpthread

all: preload.so launcher

preload.so: preload.cpp hostcall_service.cpp hostcall_service.h ../hostcalls.h
	$(CXX) -shared -fPIC $(CXXFLAGS) preload.cpp hostcall_service.cpp -o $@ $(LDFLAGS) -ldl

launcher: launcher.cpp hostcall_service.cpp hostcall_service.h ../hostcalls.h
	$(CXX) $(CXXFLAGS) launcher.cpp hostcall_service.cpp -o $@ $(LDFLAGS)

clean:
	rm -f preload.so launcher
