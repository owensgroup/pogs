#include <random>
#include <vector>
#include <cstdlib>

#include <mpi.h>

#include "schedule.h"
#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
double Lasso(pogs::Schedule &s, size_t m, size_t n, int seed) {
  std::vector<T> A;
  std::vector<T> b;
  T lambda_max;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  MASTER(kRank) {
    A.resize(m * n);
    b.resize(m);

    printf("seed: %d\n", seed);
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                             static_cast<T>(1));
    std::normal_distribution<T> n_dist(static_cast<T>(0),
                                       static_cast<T>(1));


    for (unsigned int i = 0; i < m * n; ++i)
      A[i] = n_dist(generator);

    std::vector<T> x_true(n);
    for (unsigned int i = 0; i < n; ++i)
      x_true[i] = u_dist(generator) < static_cast<T>(0.8)
                                      ? static_cast<T>(0)
                                      : n_dist(generator) /
                                      static_cast<T>(std::sqrt(n));

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < m; ++i)
      for (unsigned int j = 0; j < n; ++j)
        b[i] += A[i * n + j] * x_true[j];
    // b[i] += A[i + j * m] * x_true[j];

    for (unsigned int i = 0; i < m; ++i)
      b[i] += static_cast<T>(0.5) * n_dist(generator);

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

  MPI_Barrier(MPI_COMM_WORLD);

  pogs::MatrixDistDense<T> A_(s, 'r', m, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDistDense<T> > pogs_data(A_);
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

  double t = timer<double>();
  pogs_data.Solve(f, g);

  return timer<double>() - t;
}

template double Lasso<double>(pogs::Schedule &s, size_t m, size_t n, int seed);
template double Lasso<float>(pogs::Schedule &s, size_t m, size_t n, int seed);
