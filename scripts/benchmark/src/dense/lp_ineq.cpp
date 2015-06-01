#include <random>
#include <vector>

#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"

// Linear program in inequality form.
//   minimize    c^T * x
//   subject to  Ax <= b.
//
// See <pogs>/matlab/examples/lp_ineq.m for detailed description.
template <typename T>
double LpIneq(pogs::Schedule &s, size_t m, size_t n, int seed) {
  std::vector<T> A;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  MASTER(kRank) {
    A.resize(m * n);

    // Generate A according to:
    //   A = [-1 / n *rand(m - n, n); -eye(n)]
    for (unsigned int i = 0; i < (m - n) * n; ++i)
      A[i] = -static_cast<T>(1) / static_cast<T>(n) * u_dist(generator);
    for (unsigned int i = static_cast<unsigned int>((m - n) * n); i < m * n; ++i)
      A[i] = (i - (m - n) * n) % (n + 1) == 0 ? -1 : 0;
  }

  pogs::MatrixDistDense<T> A_(s, 'r', m, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDistDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  MASTER(kRank) {
    // Generate b according to:
    //   b = A * rand(n, 1) + 0.2 * rand(m, 1)
    f.reserve(m);
    for (unsigned int i = 0; i < m; ++i) {
      T b_i = static_cast<T>(0);
      for (unsigned int j = 0; j < n; ++j)
        b_i += A[i * n + j] * u_dist(generator);
      b_i += static_cast<T>(0.2) * u_dist(generator);
      f.emplace_back(kIndLe0, static_cast<T>(1), b_i);
    }

    // Generate c according to:
    //   c = rand(n, 1)
    g.reserve(n);
    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kIdentity, u_dist(generator) / n);
  }

  double t = timer<double>();
  pogs_data.Solve(f, g);

  return timer<double>() - t;
}

template double LpIneq<double>(pogs::Schedule &s, size_t m, size_t n, int seed);
template double LpIneq<float>(pogs::Schedule &s, size_t m, size_t n, int seed);
