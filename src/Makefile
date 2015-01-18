# User Vars

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -Wall -Wconversion -std=c++11

# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_20 -Xcompiler -fPIC -D__STRICT_ANSI__

# MPI Flags
MPI_CFLAGS = $(shell mpic++ --showme:compile)
MPI_LFLAGS = $(shell mpic++ --showme:link)

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


