#include <limits>
#include <random>
#include <vector>

#include "schedule.h"
#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"

using namespace pogs;

template <typename T>
T MaxDiff(std::vector<T> *v1, std::vector<T> *v2) {
  T max_diff = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max : max_diff)
#endif
  for (unsigned int i = 0; i < v1->size(); ++i)
    max_diff = std::max(max_diff, std::abs((*v1)[i] - (*v2)[i]));
  return max_diff;
}

template <typename T>
T Asum(std::vector<T> *v) {
  T asum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : asum)
#endif
  for (unsigned int i = 0; i < v->size(); ++i)
    asum += std::abs((*v)[i]);
  return asum;
}

// LassoPath
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// for 50 values of \lambda.
// See <pogs>/matlab/examples/lasso_path.m for detailed description.
template <typename T>
double LassoPath(Schedule &s, size_t m, size_t n, int seed) {
  unsigned int nlambda = 100;
  std::vector<T> A;
  std::vector<T> b;
  std::vector<T> x_last(n, std::numeric_limits<T>::max());
  T lambda_max;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  MASTER(kRank) {
    A.resize(m * n);
    b.resize(m);
    // Generate data
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                             static_cast<T>(1));
    std::normal_distribution<T> n_dist(static_cast<T>(0),
                                       static_cast<T>(1));

    for (unsigned int i = 0; i < m * n; ++i)
      A[i] = n_dist(generator);

    std::vector<T> x_true(n);
    for (unsigned int i = 0; i < n; ++i)
      x_true[i] = u_dist(generator) < 0.8 ? 0 : n_dist(generator) / n;

    for (unsigned int i = 0; i < m; ++i) {
      for (unsigned int j = 0; j < n; ++j)
        b[i] += A[i * n + j] * x_true[j];
      // b[i] += A[i + j * m] * x_true[j];
      b[i] += static_cast<T>(0.5) * n_dist(generator);
    }

    lambda_max = static_cast<T>(0);
    for (unsigned int j = 0; j < n; ++j) {
      T u = 0;
      for (unsigned int i = 0; i < m; ++i)
        //u += A[i * n + j] * b[i];
        u += A[i + j * m] * b[i];
      lambda_max = std::max(lambda_max, std::abs(u));
    }
  }

  // Set up pogs datastructure.
  pogs::MatrixDistDense<T> A_(s, 'r', m, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDistDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  MASTER(kRank) {
    f.reserve(m);
    g.reserve(n);

    for (unsigned int i = 0; i < m; ++i)
      f.emplace_back(kSquare, static_cast<T>(1), b[i]);

    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kAbs);
  }

  double t = timer<double>();
  for (unsigned int i = 0; i < nlambda; ++i) {
    MASTER(kRank) {
      T lambda = std::exp((std::log(lambda_max) * (nlambda - 1 - i) +
                           static_cast<T>(1e-2) * std::log(lambda_max) * i) /
                          (nlambda - 1));

      for (unsigned int i = 0; i < n; ++i)
        g[i].c = lambda;
    }

    pogs_data.Solve(f, g);

    MASTER(kRank) {
      std::vector<T> x(n);
      for (unsigned int i = 0; i < n; ++i)
        x[i] = pogs_data.GetX()[i];

      if (MaxDiff(&x, &x_last) < 1e-3 * Asum(&x))
        break;
      x_last = x;
    }
  }

  return timer<double>() - t;
}

template double LassoPath<double>(Schedule &s, size_t m, size_t n, int seed);
template double LassoPath<float>(Schedule &s, size_t m, size_t n, int seed);
