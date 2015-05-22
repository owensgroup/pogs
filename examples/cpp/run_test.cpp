#include <iostream>
#include <fstream>
#include <streambuf>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include "boost/program_options.hpp"

#include "examples.h"

typedef double real_t;

template<typename T>
using ProblemFn = double (*)(size_t m, size_t n, int seed);

enum ProblemType {
  LASSO,
  LASSO_PATH,
  LOGISTIC,
  LP_EQ,
  LP_INEQ,
  NON_NEG_L2,
  SVM
};

template <typename T>
double ErrorProblem(Schedule&, size_t, size_t, int) {
  std::cerr << "Problem type invalid" << std::endl;
  std::exit(EXIT_FAILURE);
  return 0.0;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  int m, n, seed;
  std::string typ;

  ProblemType pType;
  ProblemFn<real_t> problem;

  // CUDA sometimes takes a long time on the first API call
  // We set device here so that we avoid measuring that setup time as
  // part of the test
  cudaSetDevice(0);

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print this help message")
    ("type", po::value<std::string>(&typ), "Type of problem to run")
    ("m", po::value<int>(&m), "# of rows in generated matrix")
    ("n", po::value<int>(&n), "# of columns in generated matrix")
    ("seed", po::value<int>(&seed), "seed");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm); // can throw

    /** --help option
     */
    if ( vm.count("help")  ) {
      std::cout << "POGS test driver" << std::endl
                << desc << std::endl;
      return EXIT_SUCCESS;
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
  } else if (typ == "lasso_path") {
    pType = LASSO_PATH;
  } else if (typ == "logistic") {
    pType = LOGISTIC;
  } else if (typ == "lp_eq") {
    pType = LP_EQ;
  } else if (typ == "lp_ineq") {
    pType = LP_INEQ;
  } else if (typ == "non_neg_l2") {
    pType = NON_NEG_L2;
  } else if (typ == "svm") {
    pType = SVM;
  } else {
    std::cout << "No problem of that type\n" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  switch(pType) {
  case LASSO:
    problem = &Lasso<real_t>;
    break;
  case LASSO_PATH:
    problem = &LassoPath<real_t>;
    break;
  case LOGISTIC:
    problem = &Logistic<real_t>;
    break;
  case LP_EQ:
    problem = &LpEq<real_t>;
    break;
  case LP_INEQ:
    problem = &LpIneq<real_t>;
    break;
  case NON_NEG_L2:
    problem = &NonNegL2<real_t>;
    break;
  case SVM:
    problem = &Svm<real_t>;
    break;
  default:
    problem = &ErrorProblem<real_t>;
    break;
  }

  double ret = problem(m, n, seed);
  if (ret != -1) {
    ret = 0;
  }


  return static_cast<int>(ret);
}
