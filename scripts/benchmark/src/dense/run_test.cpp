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

#include "timer.h"
#include "util.h"
#include "parse_schedule.h"
#include "pogs.h"
#include "matrix/matrix_dist_dense.h"
#include "examples.h"


void BcastString(std::string &s) {
  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
  size_t s_size = s.size();
  MPI_Bcast(&s_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  char *buf = new char[s_size + 1];
  MASTER(kRank) {
    memcpy(buf, s.data(), s_size*sizeof(char));
  }
  s.resize(s_size);
  MPI_Bcast(buf, s_size * sizeof(char), MPI_BYTE, 0, MPI_COMM_WORLD);
  buf[s_size] = '\0';
  s.resize(s_size);
  s.replace(0, std::string::npos, buf);
  delete [] buf;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  int m, n, seed;
  ExampleData<real_t> data;
  std::string typ;
  std::string schedule_file;
  std::string sched_string;
  std::string matrix_file;

  int kRank;

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
      ("seed", po::value<int>(&seed), "seed")
      ("matrix", po::value<std::string>(&matrix_file),
       "Binary file containing f, g, and A");

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

    std::ifstream sched_fs (schedule_file, std::fstream::in);
    sched_string = std::string((std::istreambuf_iterator<char>(sched_fs)),
                               std::istreambuf_iterator<char>());

    if (matrix_file.size() > 0) {
      real_t *a;
      LoadMatrix(matrix_file, &a, data.f, data.g);
      data.A = std::vector<real_t>(a, a + (data.f.size() * data.g.size()));
      delete[] a;
    } else {
      ProblemType pType = GetProblemFn(typ);
      GenFn<real_t> problem = ExampleFns[pType];
      double t0 = timer<double>();
      data = problem(m, n, seed);
      printf("time to gen matrix: %.3e\n", timer<double>() - t0);
    }
    m = data.f.size();
    n = data.g.size();
  }

  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);


  // Sched
  BcastString(sched_string);

  Schedule sched = parse_schedule(sched_string.data(), m, n);

  pogs::MatrixDistDense<real_t> A_(sched, 'r', m, n, data.A.data());
  pogs::PogsDirect<real_t, pogs::MatrixDistDense<real_t> > pogs_data(A_);

  double ret = pogs_data.Solve(data.f, data.g);

  MPI_Finalize();

  return static_cast<int>(ret);
}
