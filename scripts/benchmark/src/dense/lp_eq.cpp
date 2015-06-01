#include <random>
#include <vector>

#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"

// Linear program in equality form.
//   minimize    c^T * x
//   subject to  Ax = b
//               x >= 0.
//
// See <pogs>/matlab/examples/lp_eq.m for detailed description.
template <typename T>
double LpEq(pogs::Schedule &s, size_t m, size_t n, int seed) {
  std::vector<T> A;

  m = m - 1;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  MASTER(kRank) {
    A.resize((m + 1) * n);


    // Generate A and c according to:
    //   A = 1 / n * rand(m, n)
    //   c = 1 / n * rand(n, 1)
    for (unsigned int i = 0; i < (m + 1) * n; ++i)
      A[i] = u_dist(generator) / static_cast<T>(n);
  }

  pogs::MatrixDistDense<T> A_(s, 'r', m + 1, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDistDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  // Generate b according to:
  //   v = rand(n, 1)
  //   b = A * v
  MASTER(kRank) {
    std::vector<T> v(n);
    for (unsigned int i = 0; i < n; ++i)
      v[i] = u_dist(generator);

    f.reserve(m + 1);
    for (unsigned int i = 0; i < m; ++i) {
      T b_i = static_cast<T>(0);
      for (unsigned int j = 0; j < n; ++j)
        b_i += A[i * n + j] * v[j];
      f.emplace_back(kIndEq0, static_cast<T>(1), b_i);
    }
    f.emplace_back(kIdentity);

    g.reserve(n);
    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kIndGe0);
  }

  double t = timer<double>();
  pogs_data.Solve(f, g);

  return timer<double>() - t;
}

template double LpEq<double>(pogs::Schedule &s, size_t m, size_t n, int seed);
template double LpEq<float>(pogs::Schedule &s, size_t m, size_t n, int seed);