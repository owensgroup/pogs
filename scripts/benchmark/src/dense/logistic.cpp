#include <random>
#include <vector>

#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"

using namespace pogs;

// Logistic
//   minimize    \sum_i -d_i y_i + log(1 + e ^ y_i) + \lambda ||x||_1
//   subject to  y = Ax
//
// for 50 values of \lambda.
// See <pogs>/matlab/examples/logistic_regression.m for detailed description.
template <typename T>
double Logistic(Schedule &s, size_t m, size_t n, int seed) {
  std::vector<T> A;
  std::vector<T> d;
  T lambda_max;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  MASTER(kRank) {
    A.resize(m * (n + 1));
    d.resize(m);

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                             static_cast<T>(1));
    std::normal_distribution<T> n_dist(static_cast<T>(0),
                                       static_cast<T>(1));

    for (unsigned int i = 0; i < m; ++i) {
      for (unsigned int j = 0; j < n; ++j)
        A[i * (n + 1) + j] = n_dist(generator);
      A[i * (n + 1) + n] = 1;
    }

    std::vector<T> x_true(n + 1);
    for (unsigned int i = 0; i < n; ++i)
      x_true[i] = u_dist(generator) < 0.8 ? 0 : n_dist(generator) / n;
    x_true[n] = n_dist(generator) / n;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < m; ++i) {
      d[i] = 0;
      for (unsigned int j = 0; j < n + 1; ++j)
        // u += A[i + j * m] * x_true[j];
        d[i] += A[i * n + j] * x_true[j];
    }
    for (unsigned int i = 0; i < m; ++i)
      d[i] = 1 / (1 + std::exp(-d[i])) > u_dist(generator);

    lambda_max = static_cast<T>(0);
#ifdef _OPENMP
#pragma omp parallel for reduction(max : lambda_max)
#endif
    for (unsigned int j = 0; j < n; ++j) {
      T u = 0;
      for (unsigned int i = 0; i < m; ++i)
        // u += A[i * n + j] * (static_cast<T>(0.5) - d[i]);
        u += A[i + j * m] * (static_cast<T>(0.5) - d[i]);
      lambda_max = std::max(lambda_max, std::abs(u));
    }
  }

  pogs::MatrixDistDense<T> A_(s, 'r', m, n + 1, A.data());
  pogs::PogsDirect<T, pogs::MatrixDistDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  MASTER(kRank) {
    f.reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      f.emplace_back(kLogistic, 1, 0, 1, -d[i]);

    g.reserve(n + 1);
    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);
    g.emplace_back(kZero);
  }

  double t = timer<double>();
  pogs_data.Solve(f, g);

  return timer<double>() - t;
}

template double Logistic<double>(Schedule &s, size_t m, size_t n, int seed);
template double Logistic<float>(Schedule &s, size_t m, size_t n, int seed);
