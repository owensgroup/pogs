#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "examples.h"
#include "util.h"

typedef double real_t;

template<typename T>
using ProblemFn = double (*)(int m_nodes, int m, int n, int nnz);

enum ProblemType {
  LASSO,
  LP
};

inline int parse_int_arg(const char *arg, const char *errmsg) {
  int out;
  char *end;

  out = static_cast<int>(strtol(arg, &end, 10));
  if (end == arg) {
    std::cerr << errmsg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return out;
}

inline bool is_problem(const char *typ, const char *problem) {
  return strcmp(typ, problem) == 0;
}

template <typename T>
double ErrorProblem(int, int, int, int) {
  std::cerr << "Problem type invalid" << std::endl;
  std::exit(EXIT_FAILURE);
  return 0.0;
}

int main(int argc, char **argv) {
  int m_nodes, m, n, nnz;
  char *typ;
  int kRank;
  ProblemType pType;
  ProblemFn<real_t> problem;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  // CUDA sometimes takes a long time on the first API call
  // We set device here so that we avoid measuring that setup time as
  // part of the test
  cudaSetDevice(0);

  MASTER(kRank) {
    if (argc < 6) {
      std::cout << "Not enough arguments" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    typ = argv[2];
    m_nodes = parse_int_arg(argv[1], "Can't convert m_nodes arg to int");
    m = parse_int_arg(argv[3], "Can't convert m arg to int");
    n = parse_int_arg(argv[4], "Can't convert n arg to int");
    nnz = parse_int_arg(argv[5], "Can't convert nnz arg to int");

    if (is_problem(typ, "lasso")) {
      pType = LASSO;
    } else if (is_problem(typ, "lp")) {
      pType = LP;
    } else {
      std::cout << "No problem of that type\n" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  MPI_Bcast(&pType, 1, MPI_INT, 0, MPI_COMM_WORLD);

  switch(pType) {
  case LASSO:
    problem = &Lasso<real_t>;
    break;
  case LP:
    problem = &LpEq<real_t>;
    break;
  default:
    problem = &ErrorProblem<real_t>;
    break;
  }

  problem(m_nodes, m, n, nnz);

  MPI_Finalize();
  
  return 0;
}
