#include <random>
#include <vector>

#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"

using namespace pogs;

// Support Vector Machine.
//   minimize    (1/2) ||w||_2^2 + \lambda \sum (a_i^T * [w; b] + 1)_+.
//
// See <pogs>/matlab/examples/svm.m for detailed description.
template <typename T>
double Svm(pogs::Schedule &s, size_t m, size_t n, int seed) {
  std::vector<T> A;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  MASTER(kRank) {
    A.resize(m * (n + 1));

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                             static_cast<T>(1));
    std::normal_distribution<T> n_dist(static_cast<T>(0),
                                       static_cast<T>(1));

    // Generate A according to:
    //   x = [randn(N, n) + ones(N, n); randn(N, n) - ones(N, n)]
    //   y = [ones(N, 1); -ones(N, 1)]
    //   A = [(-y * ones(1, n)) .* x, -y]
    for (unsigned int i = 0; i < m; ++i) {
      T sign_yi = i < m / 2 ? static_cast<T>(1) :
        static_cast<T>(-1);
      for (unsigned int j = 0; j < n; ++j) {
        A[i * (n + 1) + j] = -sign_yi * (n_dist(generator) + sign_yi);
      }
      A[i * (n + 1) + n] = -sign_yi;
    }
  }

  pogs::MatrixDistDense<T> A_(s, 'r', m, n + 1, A.data());
  pogs::PogsDirect<T, pogs::MatrixDistDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  MASTER(kRank) {
    T lambda = static_cast<T>(1);

    f.reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      f.emplace_back(kMaxPos0, static_cast<T>(1),
                     static_cast<T>(-1), lambda);

    g.reserve(n + 1);
    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kSquare);
    g.emplace_back(kZero);
  }

  double t = timer<double>();
  pogs_data.Solve(f, g);

  return timer<double>() - t;
}

template double Svm<double>(pogs::Schedule &s, size_t m, size_t n, int seed);
template double Svm<float>(pogs::Schedule &s, size_t m, size_t n, int seed);
