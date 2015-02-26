#include <random>
#include <vector>
#include <mpi.h>

#include "mat_gen.h"
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
double LpEq(int m_nodes, int m, int n, int nnz) {
  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  std::vector<T> val;
  std::vector<int> col_ind;
  std::vector<int> row_ptr;
  std::vector<T> x(n);
  std::vector<T> y(m + 1);

  std::default_random_engine generator;
  generator.seed(12410);
  srand(1240);
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));

  MASTER(kRank) {
    val.resize(nnz);
    col_ind.resize(nnz);
    row_ptr.resize(m + 2);


    // Enforce c == rand(n, 1)
    std::vector<std::tuple<int, int, T>> entries;
    entries.reserve(n);
    for (int i = 0; i < n; ++i) {
      entries.push_back(std::make_tuple(m, i, u_dist(generator)));
    }

    // Generate A and c according to:
    //   A = 4 / n * rand(m, n)
    nnz = MatGenApprox(m + 1, n, nnz, val.data(), row_ptr.data(), col_ind.data(),
                       static_cast<T>(0), static_cast<T>(4.0 / n), entries);
  }
  
  Sparse<T, int, ROW> A_(val.data(), row_ptr.data(), col_ind.data(), nnz);
  PogsData<T, Sparse<T, int, ROW>> pogs_data(A_, m + 1, n);
  pogs_data.x = x.data();
  pogs_data.m_nodes = m_nodes;
  pogs_data.n_nodes = 1;
  pogs_data.max_iter = 1000000;

  // Generate b according to:
  //   v = rand(n, 1)
  //   b = A * v
  MASTER(kRank) {
    std::vector<T> v(n);
    for (unsigned int i = 0; i < n; ++i)
      v[i] = u_dist(generator);

    pogs_data.f.reserve(m + 1);
    for (unsigned int i = 0; i < m; ++i) {
      T b_i = static_cast<T>(0);
      for (unsigned int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
        b_i += val[j] * v[col_ind[j]];
      pogs_data.f.emplace_back(kIndEq0, static_cast<T>(1), b_i);
    }
    pogs_data.f.emplace_back(kIdentity);

    pogs_data.g.reserve(n);
    for (unsigned int i = 0; i < n; ++i)
      pogs_data.g.emplace_back(kIndGe0);
  }

  double t = timer<double>();
  int error = Pogs(&pogs_data);
  t = timer<double>() - t;

  double ret;
  if (error == 0) {
    ret = t;
  } else {
    ret = -1;
  }

  return ret;
}

template double LpEq<double>(int m_nodes, int m, int n, int nnz);
template double LpEq<float>(int m_nodes, int m, int n, int nnz);

