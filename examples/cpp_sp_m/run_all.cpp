#include <cstdio>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "examples.h"
#include "util.h"

typedef double real_t;

int main(int argc, char **argv) {
  int m_nodes = 4;
  int kRank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
  
  if (argc == 2) {
    m_nodes = atoi(argv[1]);
  }

  // CUDA sometimes takes a long time on the first API call
  // We set device here so that we avoid measuring that setup time as
  // part of the measurement
  cudaSetDevice(0);

  double t;
  MASTER(kRank) {
    printf("\nLasso.\n");
  }
  t = Lasso<real_t>(m_nodes, 1000, 10000, 100000);
//  t = Lasso<real_t>(10000, 10000000, 200000000);
  MASTER(kRank) {
    printf("Solver Time: %e sec\n", t);
  }

  MASTER(kRank) {
    printf("\nLasso Path.\n");
  }
  //t = LassoPath<real_t>(200, 1000, 10000, m_nodes);
  MASTER(kRank) {
    printf("Solver Time: %e sec\n", t);
  }

//   printf("\nLogistic Regression.\n");
//   t = Logistic<real_t>(1000, 100);
//   printf("Solver Time: %e sec\n", t);

  MASTER(kRank) {
    printf("\nLinear Program in Equality Form.\n");
  }
  //t = LpEq<real_t>(100000, 10000000, 200000000, m_nodes);
  t = LpEq<real_t>(m_nodes, 2000, 1000, 100000);
  MASTER(kRank) {
    printf("Solver Time: %e sec\n", t);
  }

//   printf("\nLinear Program in Inequality Form.\n");
//   t = LpIneq<real_t>(1000, 200);
//   printf("Solver Time: %e sec\n", t);

//   printf("\nNon-Negative Least Squares.\n");
//   t = NonNegL2<real_t>(1000, 200);
//   printf("Solver Time: %e sec\n", t);

//   printf("\nSupport Vector Machine.\n");
//   t = Svm<real_t>(1000000, 2000);
//   printf("Solver Time: %e sec\n", t);

  MPI_Finalize();
  return 0;
}

