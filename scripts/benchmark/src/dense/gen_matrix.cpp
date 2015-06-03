#include <iostream>
#include <fstream>
#include <streambuf>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include "boost/program_options.hpp"

#include "examples.h"
#include "util.h"



int main(int argc, char **argv) {
  namespace po = boost::program_options;

  int m, n, seed;
  std::string typ;
  std::string out;

  int kRank;
  ProblemType pType;
  GenFn<real_t> problem;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  // CUDA sometimes takes a long time on the first API call
  // We set device here so that we avoid measuring that setup time as
  // part of the test
  //cudaSetDevice(0);

  MASTER(kRank) {
    po::options_description desc("Options");
    desc.add_options()
      ("help", "Print this help message")
      ("type", po::value<std::string>(&typ), "Type of problem to generate")
      ("m", po::value<int>(&m), "# of rows in generated matrix")
      ("n", po::value<int>(&n), "# of columns in generated matrix")
      ("out", po::value<std::string>(&out), "File to save matrix to")
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
    pType = GetProblemFn(typ);
  }

  MPI_Bcast(&pType, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

  problem = ExampleFns[pType];

#ifdef _OPENMP
  printf("openmp is working\n");
  printf("openmp max threads: %d\n", omp_get_max_threads());
#endif

  ExampleData<real_t> data = problem(m, n, seed);

  SaveMatrix(data.A.data(), data.f, data.g, out);

  MPI_Finalize();

  return 0;
}
