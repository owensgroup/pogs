#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "boost/program_options.hpp"

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

template <typename T>
double ErrorProblem(int, int, int, int) {
  std::cerr << "Problem type invalid" << std::endl;
  std::exit(EXIT_FAILURE);
  return 0.0;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  int m_nodes, m, n, nnz;
  std::string typ;
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
    po::options_description desc("Options"); 
    desc.add_options() 
      ("help", "Print this help message") 
      ("type", po::value<std::string>(&typ), "Type of problem to run")
      ("m_nodes", po::value<int>(&m_nodes),
       "Number of nodes to row split across") 
      ("m", po::value<int>(&m), "# of rows in generated matrix")
      ("n", po::value<int>(&n), "# of columns in generated matrix")
      ("nnz", po::value<int>(&nnz), "# of nonzeros in generated matrix");
 
    po::variables_map vm; 
    try { 
      po::store(po::parse_command_line(argc, argv, desc), vm); // can throw 
 
      /** --help option 
       */ 
      if ( vm.count("help")  ) { 
        std::cout << "POGS test driver" << std::endl 
                  << desc << std::endl; 
        return SUCCESS; 
      } 
 
      po::notify(vm); // throws on error, so do after help in case 
                      // there are any problems 
    } 
    catch(po::error& e) { 
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
      std::cerr << desc << std::endl; 
      exit(EXIT_FAILURE);
    } 

    if (typ == "lasso") {
      pType = LASSO;
    } else if (typ == "lp") {
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

  double ret = problem(m_nodes, m, n, nnz);
  if (ret != -1) {
      ret = 0;
  }

  MPI_Finalize();
  
  return static_cast<int>(ret);
}
