#include <random>
#include <vector>
#include <cstdlib>
#include <string>

#include <mpi.h>

#include "examples.h"
#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
ExampleData<T> Lasso(size_t m, size_t n, int seed) {
  std::vector<T> A;
  std::vector<T> b;
  T lambda_max;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  MASTER(kRank) {
    A.resize(m * n);
    b.resize(m);

    std::uniform_real_distribution<T> u_dist_template(static_cast<T>(0),
                                                      static_cast<T>(1));
    std::normal_distribution<T> n_dist_template(static_cast<T>(0),
                                                static_cast<T>(1));

    std::default_random_engine generator[NUM_RANDS];
    std::uniform_real_distribution<T> u_dist[NUM_RANDS];
    std::normal_distribution<T> n_dist[NUM_RANDS];

    for (int i = 0; i < NUM_RANDS; ++i) {
      generator[i].seed(seed + i);
      u_dist[i].param(u_dist_template.param());
      n_dist[i].param(n_dist_template.param());
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(NUM_RANDS)
#endif
    for (int i = 0; i < NUM_RANDS; ++i) {
      size_t thread_m = m / NUM_RANDS;
      size_t offset = (thread_m * i) * n;
      for (size_t j = 0; j < n * thread_m; ++j) {
        A[offset + j] = n_dist[i](generator[i]);
      }
    }

    std::vector<T> x_true(n);
    for (unsigned int i = 0; i < n; ++i)
      x_true[i] = u_dist[0](generator[0]) < static_cast<T>(0.8)
                                            ? static_cast<T>(0)
                                            : n_dist[0](generator[0]) /
                                            static_cast<T>(std::sqrt(n));

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < m; ++i)
      for (unsigned int j = 0; j < n; ++j)
        b[i] += A[i * n + j] * x_true[j];
    // b[i] += A[i + j * m] * x_true[j];

    for (unsigned int i = 0; i < m; ++i)
      b[i] += static_cast<T>(0.5) * n_dist[0](generator[0]);

    lambda_max = static_cast<T>(0);
#ifdef _OPENMP
#pragma omp parallel for reduction(max : lambda_max)
#endif
    for (unsigned int j = 0; j < n; ++j) {
      T u = 0;
      for (unsigned int i = 0; i < m; ++i)
        //u += A[i * n + j] * b[i];
        u += A[i + j * m] * b[i];
      lambda_max = std::max(lambda_max, std::abs(u));
    }
  }

  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  MASTER(kRank) {
    f.reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      f.emplace_back(kSquare, static_cast<T>(1), b[i]);

    g.reserve(n);
    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(0.2) * lambda_max);
  }

  return {A, f, g};
}

template ExampleData<double> Lasso<double>(size_t m, size_t n, int seed);
template ExampleData<float> Lasso<float>(size_t m, size_t n, int seed);
