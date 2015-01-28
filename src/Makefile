# User Vars

# C++ Flags
CXX=g++-4.8
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_20 -Xcompiler -fPIC -D__STRICT_ANSI__ -ccbin $(CXX)

# MPI Flags
MPI_CFLAGS = $(shell mpic++ --showme:compile)
MPI_LFLAGS = $(shell mpic++ --showme:link)

MPI_ROOT=/usr/local/openmpi
MPI_CFLAGS=-I$(MPI_ROOT)/include 
MPI_LFLAGS=-Xlinker -rpath -Xlinker $(MPI_ROOT)/lib -Xlinker \
--enable-new-dtags -L$(MPI_ROOT)/lib -lmpi_cxx -lmpi

OMPI_CUDA=$(shell ompi_info --parsable --all | \
grep mpi_built_with_cuda_support:value:true)

ifneq ($(strip $(OMPI_CUDA)),)
MPI_CFLAGS+=-DPOGS_OMPI_CUDA
endif

all: pogs.o pogs_cu_link.o pogs_cu.o pogs_sp_m_cu.o cml.o

# CPU
pogs.o: pogs.cpp pogs.h prox_lib.h _interface_defs.h sinkhorn_knopp.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) $< -c -o $@

# GPU
pogs_cu_link.o: pogs_cu.o pogs_sp_m_cu.o cml.o
	$(CUXX) $(CUFLAGS) $(MPI_CFLAGS) $^ $(MPI_LFLAGS) -dlink -o $@

pogs_cu.o: pogs.cu pogs.h prox_lib.h
	$(CUXX) $(CUFLAGS) $(IFLAGS) $< -dc -o $@

#pogs_sp_cu.o: pogs_sp.cu pogs.h prox_lib.h
#	$(CUXX) $(CUFLAGS) $(IFLAGS) $< -dc -o $@

pogs_sp_m_cu.o: pogs_sp_m.cu pogs.h prox_lib.h
	$(CUXX) $(CUFLAGS) $(MPI_CFLAGS) $(IFLAGS) $< $(MPI_LFLAGS) -dc -o $@

cml.o: cml/cml_blas.cu cml/cml_blas.cuh
	$(CUXX) -Icml $(CUFLAGS) $< -dc -o $@

clean:
	rm -f *.o *~
	rm -rf *.dSYM


