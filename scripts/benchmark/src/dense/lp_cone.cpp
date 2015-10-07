#include <random>
#include <vector>

#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"
#include "examples.h"

// Linear program conic
//   minimize    c^T * x
//   subject to  Ax + s = b
//               s(1:N) = 0
//               s(N + 1:m) >= 0.
//
// See <pogs>/matlab/examples/lp_cone.m for detailed description.
template <typename T>
ExampleData<T> LpCone(pogs::Schedule &s, size_t m, size_t n, int seed) {
  int NUM_BLOCKS = s.MBlocks();
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

    // Let
    //   N = floor(min(m, n) / 4)
    // Generate A and c according to:
    //   A = 4 / n * rand(m, n)
    //   b = A * rand(n, 1) + [zeros(N, 1); 0.1 * rand(m - N, 1)];
    //   c = -A' * rand(m, 1)

    long N = min(m, n) / 4;

    // Generate A
#ifdef _OPENMP
#pragma omp parallel for num_threads(NUM_RANDS)
#endif
    for (int i = 0; i < NUM_RANDS; ++i) {
      size_t thread_m = m / NUM_RANDS;
      size_t offset_row = (thread_m * i);
      size_t offset = offset_row * n;
      for (size_t j = 0; j < n * thread_m; ++j) {
        A[offset + j] = static_cast<T>(4) / n * u_dist[i](generator[i]);
      }
    }

    // Generate b according to:
    //   v = rand(n, 1)
    //   h = [zeros(N, 1); 0.1 * rand(m - N, 1)];
    //   b = A * v + h
    std::vector<T> v(n);
    for (unsigned int i = 0; i < n; ++i)
      v[i] = u_dist[0](generator[0]);

    std::vector<T> h(m);
    for (unsigned int i = 0; i < N; ++i)
      h[i] = 0;
    for (unsigned int i = N; i < m; ++i)
      h[i] = static_cast<T>(0.1) * u_dist[0](generator[0]);

    f.reserve(m);
    for (unsigned int i = 0; i < m; ++i) {
      T b_i = h[i];
      for (unsigned int j = 0; j < n; ++j)
        b_i += A[i * n + j] * v[j];
      f.emplace_back(i < N : kIndEq0 ? kIndLe0,
                     static_cast<T>(1), b_i);
    }

    // Generate c according to:
    //   c = -A' * rand(m, 1)
    std::vector<T> c(m);
    for (unsigned int i = 0; i < m; ++i)
      c[i] = u_dist[0](generator[0]);

    g.reserve(n);
    for (unsigned int i = 0; i < n; ++i) {
      T c_i = static_cast<T>(0);
      for (unsigned int j = 0; j < m; ++j)
        c_i += -A[n * j + i] * c[j];
      g.emplace_back(kIdentity, static_cast<T>(c_i));
    }
  }

  return {A, f, g};
}

template ExampleData<double> LpCone<double>(pogs::Schedule &s, size_t m,
                                            size_t n,
                                            int seed);
template ExampleData<float> LpCone<float>(pogs::Schedule &s, size_t m, size_t n,
                                          int seed);
