#include <random>
#include <vector>
#include <string>

#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"
#include "examples.h"

// Linear program in inequality form.
//   minimize    c^T * x
//   subject to  Ax <= b.
//
// See <pogs>/matlab/examples/lp_ineq.m for detailed description.
template <typename T>
ExampleData<T> LpIneq(size_t m, size_t n, int seed) {
  std::vector<T> A;
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  MASTER(kRank) {
    A.resize(m * n);

    std::uniform_real_distribution<T> u_dist_template(static_cast<T>(0),
                                                      static_cast<T>(1));

    std::default_random_engine generator[NUM_RANDS];
    std::uniform_real_distribution<T> u_dist[NUM_RANDS];
    for (int i = 0; i < NUM_RANDS; ++i) {
      generator[i].seed(seed + i);
      u_dist[i].param(u_dist_template.param());
    }

    // Generate A according to:
    //   A = [-1 / n *rand(m - n, n); -eye(n)]
#ifdef _OPENMP
#pragma omp parallel for num_threads(NUM_RANDS)
#endif
    for (int j = 0; j < NUM_RANDS; ++j) {
      size_t thread_n = n / NUM_RANDS;
      size_t offset = (m - n) * thread_n * j;
      for (unsigned int i = 0; i < (m - n) * thread_n; ++i)
        A[offset + i] =
          -static_cast<T>(1) / static_cast<T>(n) * u_dist[j](generator[j]);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = static_cast<unsigned int>((m - n) * n); i < m * n; ++i)
      A[i] = (i - (m - n) * n) % (n + 1) == 0 ? -1 : 0;

    // Generate b according to:
    //   b = A * rand(n, 1) + 0.2 * rand(m, 1)
    f.reserve(m);
    for (unsigned int i = 0; i < m; ++i) {
      T b_i = static_cast<T>(0);
      for (unsigned int j = 0; j < n; ++j)
        b_i += A[i * n + j] * u_dist[0](generator[0]);
      b_i += static_cast<T>(0.2) * u_dist[0](generator[0]);
      f.emplace_back(kIndLe0, static_cast<T>(1), b_i);
    }

    // Generate c according to:
    //   c = rand(n, 1)
    g.reserve(n);
    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kIdentity, u_dist[0](generator[0]) / n);
  }

  return {A, f, g};
}

template ExampleData<double> LpIneq<double>(size_t m, size_t n, int seed);
template ExampleData<float> LpIneq<float>(size_t m, size_t n, int seed);
