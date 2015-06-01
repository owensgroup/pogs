#include <iostream>
#include <fstream>
#include <streambuf>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "boost/program_options.hpp"

#include "examples.h"
#include "util.h"
#include "parse_schedule.h"

typedef double real_t;

template<typename T>
using ProblemFn = double (*)(Schedule &s, size_t m, size_t n, int seed);

enum ProblemType {
  LASSO,
  LASSO_PATH,
  LOGISTIC,
  LP_EQ,
  LP_INEQ,
  NON_NEG_L2,
  SVM
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
double ErrorProblem(Schedule&, size_t, size_t, int) {
  std::cerr << "Problem type invalid" << std::endl;
  std::exit(EXIT_FAILURE);
  return 0.0;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  int m, n, seed;
  std::string typ;
  std::string schedule_file;
  std::string sched_string;

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
      ("schedule", po::value<std::string>(&schedule_file),
       "JSON file for schedule description")
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
    std::ifstream sched_fs (schedule_file, std::fstream::in);
    sched_string = std::string((std::istreambuf_iterator<char>(sched_fs)),
                               std::istreambuf_iterator<char>());
  }

  MPI_Bcast(&pType, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

  size_t sched_size = sched_string.size();
  MPI_Bcast(&sched_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  char *buf = new char[sched_size + 1];
  MASTER(kRank) {
    memcpy(buf, sched_string.data(), sched_size*sizeof(char));
  }
  sched_string.resize(sched_size);
  MPI_Bcast(buf, sched_size * sizeof(char), MPI_BYTE, 0, MPI_COMM_WORLD);
  buf[sched_size] = '\0';
  sched_string.resize(sched_size);
  sched_string.replace(0, std::string::npos, buf);
  delete [] buf;

  Schedule sched = parse_schedule(sched_string.data(), m, n);

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

  double ret = problem(sched, m, n, seed);
  if (ret != -1) {
      ret = 0;
  }

  MPI_Finalize();

  return static_cast<int>(ret);
}
