#include <random>
#include <vector>

#include "matrix/matrix_dist_dense.h"
#include "pogs.h"
#include "timer.h"
#include "util.h"
#include "examples.h"

#define NUM_BLOCKS 4

// Linear program in equality form.
//   minimize    c^T * x
//   subject to  Ax = b
//               x >= 0.
//
// See <pogs>/matlab/examples/lp_eq.m for detailed description.
template <typename T>
ExampleData<T> LpEqM(size_t m, size_t n, int seed) {
  std::vector<T> A;

  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  size_t real_m = m + NUM_BLOCKS;

  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  MASTER(kRank) {
    A.resize(real_m * n);

    std::uniform_real_distribution<T> u_dist_template(static_cast<T>(0),
                                                      static_cast<T>(1));

    std::default_random_engine generator[NUM_RANDS];
    std::uniform_real_distribution<T> u_dist[NUM_RANDS];

    for (int i = 0; i < NUM_RANDS; ++i) {
      generator[i].seed(seed + i);
      u_dist[i].param(u_dist_template.param());
    }


    // Generate A and c according to:
    //   A = 1 / n * rand(m, n)
    //   c = 1 / n * rand(n, 1)

#ifdef _OPENMP
#pragma omp parallel for num_threads(NUM_RANDS)
#endif
    for (int i = 0; i < NUM_RANDS; ++i) {
      size_t thread_m = real_m  / NUM_RANDS;
      size_t offset = (thread_m * i) * n;
      for (size_t j = 0; i < n * thread_m; ++i) {
        size_t row = (offset + j) / n ;
        if (row + 1 % (real_m / NUM_BLOCKS) == 0 && row != real_m - 1)
          continue;
        A[offset + j] = u_dist[i](generator[i]) / static_cast<T>(n);
      }
    }

    size_t c_start = real_m * n - 1;
    for (size_t i = 0; i < NUM_BLOCKS - 1; ++i) {
      size_t c_size = (n / NUM_BLOCKS);
      size_t c_offset = c_size * (i - 1);
      size_t row = (m / NUM_BLOCKS) * i + i;
      for (size_t j = 0; j < n; ++j) {
        A[row * n + j] = 0;
      }
      for (size_t j = 0; j < c_size; ++j) {
        A[row * n + c_offset + j] = A[c_start + c_offset + j];
        A[c_start + c_offset + j] = 0;
      }
    }

  // Generate b according to:
  //   v = rand(n, 1)
  //   b = A * v
    std::vector<T> v(n);
    for (unsigned int i = 0; i < n; ++i)
      v[i] = u_dist[0](generator[0]);

    f.reserve(real_m);
    for (unsigned int i = 0; i < real_m; ++i) {
      if (i + 1 % (real_m / NUM_BLOCKS) == 0) {
        f.emplace_back(kIdentity);
      }
      T b_i = static_cast<T>(0);
      for (unsigned int j = 0; j < n; ++j)
        b_i += A[i * n + j] * v[j];
      f.emplace_back(kIndEq0, static_cast<T>(1), b_i);
    }

    g.reserve(n);
    for (unsigned int i = 0; i < n; ++i)
      g.emplace_back(kIndGe0);
  }

  return {A, f, g};
}

template ExampleData<double> LpEqM<double>(size_t m, size_t n, int seed);
template ExampleData<float> LpEqM<float>(size_t m, size_t n, int seed);
