# Makefile for DAE kernel (multi-file build)

# CUDA compiler
NVCC = nvcc

# CUDA architecture, pass on the command line, e.g. `make pyext arch=120a`
#   90a  : GH200 (Hopper)
#   120a : RTX PRO 6000 (Blackwell)
# DAK_SM_ARCH selects the arch-specific code in include/task/config.cuh and include/task/gemv.cuh
arch ?= 90a
SUPPORTED_ARCHS := 90a 120a
ifeq ($(filter $(arch),$(SUPPORTED_ARCHS)),)
$(error Unsupported arch '$(arch)', expected one of: $(SUPPORTED_ARCHS))
endif
CUDA_ARCH = -gencode arch=compute_$(arch),code=sm_$(arch) -DDAK_SM_ARCH=$(subst a,,$(arch))

# Rewritten only when arch changes, so objects built for another arch get rebuilt
ARCH_STAMP := .build_arch

# Compiler flags
# NVCC_FLAGS = -DNDEBUG -O3 -std=c++20 $(if $(profile),-DDAE_PROFILE) # --ptxas-options=--verbose

# Linker flags (add CUDA driver library for TMA support)
LDFLAGS = -lcuda -lcublas




# Directories
ifeq ($(debug),true)
	NVCC_FLAGS = -O3 -Iinclude/offload -Iinclude -std=c++20 -DDAE_DEBUG_PRINT -Xptxas=-v
else
	NVCC_FLAGS = -O3 -Iinclude/offload -Iinclude -std=c++20 -DNDEBUG -Xptxas=-v
endif

TARGETS := runtime.o

# Target executable
# CUFILES := $(wildcard app/*.cu)
# APPS := $(patsubst app/%.cu,%,$(CUFILES))

# # Source files
# SOURCES = main.cu 

# Header files (for dependency tracking)
# HEADERS = $(wildcard include/dae/*.cuh) $(wildcard include/task/*.cuh) $(wildcard include/dae/pipeline/*.cuh)
HEADERS = $(wildcard include/offload/*.cuh) $(wildcard include/task/*.cuh)

# for make <target> run
# BIN ?= $(firstword $(filter-out run,$(MAKECMDGOALS)))

# Default target
# all: $(APPS)

# Clean build artifacts
clean:
# 	rm -rf $(APPS) $(TARGETS)
	rm -rf $(TARGETS) $(ARCH_STAMP)

$(ARCH_STAMP): FORCE
	@echo '$(arch)' | cmp -s - $@ || echo '$(arch)' > $@

%.o: src/%.cu $(HEADERS) $(ARCH_STAMP)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $@ $<

# Build the executable, this is wildcard rule for multiple targets
%: app/%.cu $(TARGETS) $(HEADERS) $(ARCH_STAMP)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -o $@ $< $(TARGETS) $(LDFLAGS)

# run: $(BIN)
# 	./$<

# setup.py reads the arch from DAK_ARCH
pyext: $(TARGETS)
	DAK_ARCH=$(arch) pip install -e . --no-build-isolation

.PHONY: all clean run pyext FORCE
