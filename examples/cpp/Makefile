# User Vars
POGSROOT=../../src
CUDAROOT=/usr/local/cuda-6.5

# Example Files
EXSRC=lasso.cpp lasso_path.cpp logistic.cpp lp_eq.cpp lp_ineq.cpp nonneg_l2.cpp svm.cpp

LD_BOOST=-I/home/abpoms/boost/boost_1_57_0
LIB_BOOST=-L/home/abpoms/boost/boost_1_57_0/stage/lib -lboost_program_options

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -O3 -I$(POGSROOT)/include $(LIB_BOOST) -std=c++11 -Wall -Wconversion

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse -I$(CUDAROOT)/include -I./include $(LD_BOOST) $(LIB_BOOST)

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L$(CUDAROOT)/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm -lopenblas
CULDFLAGS=-L$(CUDAROOT)/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) -o run $(EXSRC) $<	$(POGSROOT)/build/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

