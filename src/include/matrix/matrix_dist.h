#ifndef MATRIX_MATRIX_DIST_H_
#define MATRIX_MATRIX_DIST_H_

#include <mpi.h>
#include <memory>
#include "schedule.h"

namespace pogs {

namespace {
  const ProcessInfo& meta_init(const Schedule &s) {
    int kRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
    return s.At(kRank);
  }
}

template <typename T>
class MatrixDist {
 protected:
  const Schedule _S;
  const ProcessInfo &_meta;

  const size_t _m, _n;

  void *_info;

  bool _done_init;

 public:
  MatrixDist(const Schedule &s, size_t m, size_t n)
    : _S(s), _meta(meta_init(s)), _m(m), _n(n), _info(0), _done_init(false)
    {};

  virtual ~MatrixDist() { };

  // Call this methods to initialize the matrix.
  virtual int Init() = 0;

  // Method to equilibrate and return equilibration vectors.
  virtual int Equil(T *d, T *e) = 0;

  // Method to multiply by block A and block A^T.
  virtual int BlockMul(char trans, T alpha, const T *x, T beta, T *y) const = 0;
  // Method to multiply by A and A^T.
  virtual int Mul(char trans, T alpha, const T *x, T beta, T *y) const = 0;

  // Get dimensions and check if initialized
  size_t Rows() const { return _m; }
  size_t Cols() const { return _n; }
  size_t BlockRows() const { return _meta.block.Rows(); }
  size_t BlockCols() const { return _meta.block.Cols(); }
  bool IsInit() const { return _done_init; }
  const Schedule& Schedule() const { return _S; }
  const ProcessInfo& Meta() const { return _meta; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_DIST_H_
