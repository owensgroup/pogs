# User Vars
POGSROOT=../../src
CUDAROOT=/usr/local/cuda-6.5

# Example Files
EXSRC=lasso.cpp lp_eq.cpp lasso_path.cpp # logistic.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

LD_BOOST=-I/home/abpoms/boost/boost_1_57_0
LIB_BOOST=-L/home/abpoms/boost/boost_1_57_0/stage/lib -lboost_program_options

# C++ Flags
CXX=g++-4.8
CXXFLAGS=-g -O3 $(LD_BOOST) -I$(POGSROOT) $(LIB_BOOST) -std=c++11 -Wall -Wconversion  

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse -I$(CUDAROOT)/include
CUXXFLAGS=-D__CUDA

# MPI Flags
MPI_CFLAGS = $(shell mpic++ --showme:compile)
MPI_LFLAGS = $(shell mpic++ --showme:link)

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
CULDFLAGS=-L$(CUDAROOT)/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm
CULDFLAGS=-L$(CUDAROOT)/lib64 $(CULDFLAGS_)
endif

# CPU
# cpu: run_all.cpp examples.h $(EXSRC)
# 	$(MAKE) -C $(POGSROOT) pogs.o
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OOFLAGS) -o run $(EXSRC) $< $(POGSROOT)/pogs.o

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) -C $(POGSROOT) pogs_sp_m_cu.o pogs_cu.o pogs_cu_link.o cml.o
	$(CXX) $(CXXFLAGS) $(CUXXFLAGS) $(MPI_CFLAGS) -o run $(EXSRC) $< \
 $(POGSROOT)/pogs_sp_m_cu.o $(POGSROOT)/pogs_cu_link.o \
$(POGSROOT)/pogs_cu.o $(POGSROOT)/cml.o $(CULDFLAGS) $(MPI_LFLAGS)

test: CXXFLAGS+=-DPOGS_TEST
test: run_test.cpp examples.h $(EXSRC)
	$(MAKE) -C $(POGSROOT) test pogs_sp_m_cu.o pogs_cu.o pogs_cu_link.o cml.o
	$(CXX) $(CXXFLAGS) $(CUXXFLAGS) $(MPI_CFLAGS) -o run_test $(EXSRC) $< \
 $(POGSROOT)/pogs_sp_m_cu.o $(POGSROOT)/pogs_cu_link.o \
$(POGSROOT)/pogs_cu.o $(POGSROOT)/cml.o $(CULDFLAGS) $(MPI_LFLAGS)

clean:
	rm -f $(POGSROOT)/*.o 
	rm -f *.o *~ *~ run run_test
	rm -rf *.dSYM

