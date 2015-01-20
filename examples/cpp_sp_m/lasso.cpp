#include <random>
#include <vector>
#include <mpi.h>

#include "mat_gen.h"
#include "pogs.h"
#include "timer.h"

// Lasso
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
double Lasso(int m, int n, int nnz) {
  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  std::vector<T> val;
  std::vector<int> col_ind;
  std::vector<int> row_ptr;
  std::vector<T> b;
  std::vector<T> x(n);
  std::vector<T> y(m);
  if (kRank == 0) {
    val.resize(nnz);
    col_ind.resize(nnz);
    row_ptr.resize(m + 1);
    b.resize(m);

    std::default_random_engine generator;
    std::normal_distribution<T> n_dist(static_cast<T>(0),
                                       static_cast<T>(1));
 
    std::vector<std::tuple<int, int, T>> entries;
    nnz = MatGenApprox(m, n, nnz, val.data(), row_ptr.data(), col_ind.data(),
                       static_cast<T>(-1), static_cast<T>(1), entries);

    for (unsigned int i = 0; i < m; ++i)
      b[i] = static_cast<T>(4) * n_dist(generator);

  }

  T lambda_max = 5;

  Sparse<T, int, ROW> A_(val.data(), row_ptr.data(), col_ind.data(), nnz);
  PogsData<T, Sparse<T, int, ROW>> pogs_data(A_, m, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();
  pogs_data.m_nodes = 2;
  pogs_data.n_nodes = 1;

  if (kRank == 0) {
    pogs_data.f.reserve(m);
    for (unsigned int i = 0; i < m; ++i)
      pogs_data.f.emplace_back(kSquare, static_cast<T>(1), b[i]);

    pogs_data.g.reserve(n);
    for (unsigned int i = 0; i < n; ++i)
      pogs_data.g.emplace_back(kAbs, static_cast<T>(0.5) * lambda_max);
  }

  double t = timer<double>();
  Pogs(&pogs_data);

  return timer<double>() - t;
}

template double Lasso<double>(int m, int n, int nnz);
template double Lasso<float>(int m, int n, int nnz);
