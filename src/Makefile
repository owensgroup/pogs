# Instructions
# 1. To build with openmp set IFLAGS=-fopenmp

# Set vpath for build
ifeq (cpu,$(MAKECMDGOALS))
	vpath % cpu:cpu/matrix:cpu/projector
else ifeq (gpu,$(MAKECMDGOALS))
	vpath % gpu:gpu/matrix:gpu/projector
else ifeq (multi,$(MAKECMDGOALS))
	vpath % multi:multi/matrix:multi/projector
endif

# Bulid directory
OBJDIR=build

# MPI Flags
MPI_CFLAGS = $(shell mpic++ --showme:compile)
MPI_LFLAGS = $(shell mpic++ --showme:link)

#MPI_ROOT=/home/abpoms/mvapich2-2.1
MPI_ROOT=$(MPI_HOME)
#MPI_ROOT=/usr/local/openmpi
# MPI_CFLAGS=-I$(MPI_ROOT)/include 
# MPI_LFLAGS=-Xlinker -rpath -Xlinker $(MPI_ROOT)/lib -Xlinker \
# --enable-new-dtags -L$(MPI_ROOT)/lib -lmpi_cxx -lmpi

# C++ Flags
CXX=mpic++
CXXFLAGS=$(IFLAGS) -g -O3 -Wall -std=c++11 -fPIC -DDEBUG -DMVAPICH # -Wconversion

# CUDA Flags
CUXX=nvcc
CUFLAGS=$(IFLAGS) -arch=sm_20 -Xcompiler -fPIC -DDEBUG -DMVAPICH


# POGS header files.
POGS_HDR=\
	include/interface_defs.h \
	include/pogs.h \
	include/prox_lib.h \
	include/util.h \
	include/matrix/matrix.h \
	include/matrix/matrix_dense.h \
	include/matrix/matrix_sparse.h \
	include/projector/projector_cgls.h \
	include/projector/projector_direct.h

# CPU Specific headers and object files.
GSL_HDR=\
	cpu/include/gsl/cblas.h \
	cpu/include/gsl/gsl_blas.h \
	cpu/include/gsl/gsl_linalg.h \
	cpu/include/gsl/gsl_matrix.h \
	cpu/include/gsl/gsl_rand.h \
	cpu/include/gsl/gsl_spblas.h \
	cpu/include/gsl/gsl_spmat.h \
	cpu/include/gsl/gsl_vector.h

CPU_HDR=\
	cpu/include/cgls.h \
	cpu/include/equil_helper.h \
	cpu/include/projector_helper.h
CPU_MTX_OBJ=\
	$(OBJDIR)/cpu/matrix/matrix_sparse.o \
	$(OBJDIR)/cpu/matrix/matrix_dense.o
CPU_PRJ_OBJ=\
	$(OBJDIR)/cpu/projector/projector_cgls.o \
	$(OBJDIR)/cpu/projector/projector_direct_dense.o
CPU_OBJ=$(OBJDIR)/cpu/pogs.o

# GPU Specific headers and object files.
CML_HDR=\
	gpu/include/cml/cblas.h \
	gpu/include/cml/cml_blas.cuh \
	gpu/include/cml/cml_defs.cuh \
	gpu/include/cml/cml_linalg.cuh \
	gpu/include/cml/cml_matrix.cuh \
	gpu/include/cml/cml_rand.cuh \
	gpu/include/cml/cml_spblas.cuh \
	gpu/include/cml/cml_spmat.cuh \
	gpu/include/cml/cml_utils.cuh \
	gpu/include/cml/cml_vector.cuh

GPU_HDR=\
	gpu/include/cgls.cuh \
	gpu/include/equil_helper.cuh \
	gpu/include/projector_helper.cuh
GPU_MTX_OBJ=\
	$(OBJDIR)/gpu/matrix/matrix_dense.o \
	$(OBJDIR)/gpu/matrix/matrix_sparse.o
GPU_PRJ_OBJ=\
	$(OBJDIR)/gpu/projector/projector_cgls.o \
	$(OBJDIR)/gpu/projector/projector_direct_dense.o
GPU_OBJ=$(OBJDIR)/gpu/pogs.o

# Multi-GPU Specific headers and object fiels.
MULTI_CML_HDR=\
    include/matrix/matrix_dist.h \
    include/matrix/matrix_dist_dense.h \
	multi/include/cml/cblas.h \
	multi/include/cml/cml_blas.cuh \
	multi/include/cml/cml_defs.cuh \
	multi/include/cml/cml_linalg.cuh \
	multi/include/cml/cml_matrix.cuh \
	multi/include/cml/cml_rand.cuh \
	multi/include/cml/cml_spblas.cuh \
	multi/include/cml/cml_spmat.cuh \
	multi/include/cml/cml_utils.cuh \
	multi/include/cml/cml_vector.cuh

MULTI_HDR=\
	multi/include/cgls.cuh \
	multi/include/equil_helper.cuh \
	multi/include/projector_helper.cuh
MULTI_MTX_OBJ=\
	$(OBJDIR)/multi/matrix/matrix_dist_dense.o
#   $(OBJDIR)/multi/matrix/matrix_sparse.o
MULTI_PRJ_OBJ=\
	$(OBJDIR)/multi/projector/projector_direct_dense.o
#   $(OBJDIR)/multi/projector/projector_cgls.o
MULTI_OBJ=$(OBJDIR)/multi/pogs.o


# Build all
cpu: $(CPU_OBJ) $(CPU_MTX_OBJ) $(CPU_PRJ_OBJ)
	ar cr $(OBJDIR)/pogs.a $^

gpu: $(OBJDIR)/pogs_link.o $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ)
	ar cr $(OBJDIR)/pogs.a $^

multi: $(OBJDIR)/pogs_multi_link.o $(MULTI_OBJ) \
 $(MULTI_MTX_OBJ) $(MULTI_PRJ_OBJ)
	ar cr $(OBJDIR)/pogs.a $^

# Directories
$(OBJDIR):
	mkdir -p $@

# Directories CPU
$(OBJDIR)/cpu: $(OBJDIR)
	mkdir -p $@ 

$(OBJDIR)/cpu/matrix: $(OBJDIR)/cpu
	mkdir -p $@ 

$(OBJDIR)/cpu/projector: $(OBJDIR)/cpu
	mkdir -p $@ 

# Directories GPU
$(OBJDIR)/gpu: $(OBJDIR)
	mkdir -p $@ 

$(OBJDIR)/gpu/matrix: $(OBJDIR)/gpu
	mkdir -p $@ 

$(OBJDIR)/gpu/projector: $(OBJDIR)/gpu
	mkdir -p $@ 

# Directories Multi-GPU
$(OBJDIR)/multi: $(OBJDIR)
	mkdir -p $@ 

$(OBJDIR)/multi/matrix: $(OBJDIR)/multi
	mkdir -p $@ 

$(OBJDIR)/multi/projector: $(OBJDIR)/multi
	mkdir -p $@ 


# POGS CPU objects
$(OBJDIR)/cpu/pogs.o: cpu/pogs.cpp $(POGS_HDR) $(GSL_HDR) | $(OBJDIR)/cpu
	$(CXX) -I include -Icpu/include $< $(CXXFLAGS) -c -o $@ 

$(OBJDIR)/cpu/matrix/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/matrix
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/projector/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/projector
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

# POGS GPU objects
$(OBJDIR)/pogs_link.o: $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) | $(OBJDIR)
	$(CUXX) $(CUFLAGS) $^ -dlink -o $@ 

$(OBJDIR)/gpu/pogs.o: gpu/pogs.cu $(POGS_HDR) $(CML_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@ 

$(OBJDIR)/gpu/matrix/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/matrix
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/projector/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/projector
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

# POGS Multi-GPU objects
$(OBJDIR)/pogs_multi_link.o: $(MULTI_OBJ) $(MULTI_MTX_OBJ) $(MULTI_PRJ_OBJ) \
| $(OBJDIR)
	$(CUXX) $(CUFLAGS) $^ -dlink -o $@

$(OBJDIR)/multi/pogs.o: multi/pogs.cu $(POGS_HDR) $(MULTI_CML_HDR) \
| $(OBJDIR)/multi
	$(CUXX) -Iinclude -Imulti/include $< $(CUFLAGS) -dc -o $@

$(OBJDIR)/multi/matrix/%.o: %.cu $(MULTI_HDR) | $(OBJDIR)/multi/matrix
	$(CUXX) -Iinclude -Imulti/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/multi/projector/%.o: %.cu $(MULTI_HDR) | $(OBJDIR)/multi/projector
	$(CUXX) -Iinclude -Imulti/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/multi/%.o: %.cu $(MULTI_HDR) | $(OBJDIR)/multi
	$(CUXX) -Iinclude -Imulti/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

clean:
	rm -rf pogs.a build/ *.dSYM
