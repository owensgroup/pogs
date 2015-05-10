#ifndef MATRIX_MATRIX_DIST_DENSE_H_
#define MATRIX_MATRIX_DIST_DENSE_H_

#include "matrix_dist.h"
#include "schedule.h"

namespace pogs {

template <typename T>
class MatrixDistDense : public MatrixDist<T> {
 public:
  enum Ord {ROW, COL};

 private:
  // TODO: This should be shared cpu/gpu pointer?
  T *_data;

  Ord _ord;

  // Get rid of assignment operator.
  MatrixDistDense<T>& operator=(const MatrixDistDense<T>& A);

 public:
  // Constructor (only sets variables)
  MatrixDistDense(const Schedule &s,
                  char ord, size_t m, size_t n, const T *data);
  MatrixDistDense(const MatrixDistDense<T>& A);
  ~MatrixDistDense();

  // Initialize matrix, call this before any other methods.
  int Init();

  // Method to equilibrate.
  int Equil(T *d, T *e);

  // Method to multiply by block A and block A^T.
  int BlockMul(char trans, T alpha, const T *x, T beta, T *y) const;
  // Method to multiply by A and A^T.
  int Mul(char trans, T alpha, const T *x, T beta, T *y) const;

  // Getters
  const T* Data() const { return _data; }
  Ord Order() const { return _ord; }
};

}  // namespace pogs

#endif  // MATRIX_MATRIX_DIST_DENSE_H_

