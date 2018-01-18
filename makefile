# Makefile for APSP 

NVCC=/usr/local/cuda/bin/nvcc
CPPFLAGS= -O3 -std=c++14

CU_SRCS :=
C_UPPER_SRCS :=
CXX_SRCS :=
C++_SRCS :=
OBJ_SRCS :=
CC_SRCS :=
ASM_SRCS :=
C_SRCS :=
S_UPPER_SRCS :=
CC_DEPS :=
C++_DEPS :=
EXECUTABLES :=
C_UPPER_DEPS :=
CXX_DEPS :=
CU_DEPS :=
CPP_DEPS :=
C_DEPS :=



CPP_SRCS := \
main.cpp

O_SRCS := \
apsp.o \
cuda_apsp.o

OBJS := \
./main.o

CPP_DEPS += \
./main.d


# Build root folder
%.o: %.cpp
	@echo 'NVCC compiler building file: $<'
	$(NVCC) $(CPPFLAGS) -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	$(NVCC) $(CPPFLAGS) --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


CU_SRCS := \
lib/cuda/cuda_apsp.cu

OBJS += \
./lib/cuda/cuda_apsp.o

CU_DEPS += \
./lib/cuda/cuda_apsp.d

# Building lib/cuda
lib/cuda/%.o: lib/cuda/%.cu
	@echo 'NVCC compiler building file: $<'
	$(NVCC) $(CPPFLAGS) -gencode arch=compute_61,code=sm_61  -odir "lib/cuda" -M -o "$(@:%.o=%.d)" "$<"
	$(NVCC) $(CPPFLAGS) --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

CPP_SRCS += \
lib/apsp.cpp

OBJS += \
./lib/apsp.o

CPP_DEPS += \
./lib/apsp.d


# Build lib
lib/%.o: lib/%.cpp
	@echo 'NVCC compiler building file: $<'
	$(NVCC) $(CPPFLAGS) -gencode arch=compute_61,code=sm_61  -odir "lib" -M -o "$(@:%.o=%.d)" "$<"
	$(NVCC) $(CPPFLAGS) --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


# All Target
all: cuda_floyd-warshall

cuda_floyd-warshall: $(OBJS) $(USER_OBJS)
	@echo 'Linker building target: $@'
	$(NCC) --cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -link -o  "cuda_floyd-warshall" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'

clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(C_UPPER_DEPS)$(CXX_DEPS)$(OBJS)$(CU_DEPS)$(CPP_DEPS)$(C_DEPS) cuda_floyd-warshall
	-@echo ' '
