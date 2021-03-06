#include <cublas_v2.h>

#include <algorithm>
#include <limits>

#include "cgls.cuh"
#include "cml/cml_blas.cuh"
#include "cml/cml_vector.cuh"
#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector_cgls.h"
#include "projector_helper.cuh"
#include "util.h"

namespace pogs {

namespace {

int kMaxIter = 100;
bool kCglsQuiet = true;

template<typename T>
struct GpuData {
  cublasHandle_t handle;
  GpuData() {
    cublasCreate(&handle);
    CUDA_CHECK_ERR();
  }
  ~GpuData() {
    cublasDestroy(handle);
    CUDA_CHECK_ERR();
  }
};

// CGLS Gemv struct for matrix multiplication.
template <typename T, typename M>
struct Gemv : cgls::Gemv<T> {
  const M& A;
  Gemv(const M& A) : A(A) { }
  int operator()(char op, const T alpha, const T *x, const T beta, T *y)
      const {
    return A.Mul(op, alpha, x, beta, y);
  }
};

}  // namespace

template <typename T, typename M>
ProjectorCgls<T, M>::ProjectorCgls(const M& A)
    : _A(A) {
  // Set GPU specific this->_info.
  GpuData<T> *info = new GpuData<T>();
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T, typename M>
ProjectorCgls<T, M>::~ProjectorCgls() {
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  delete info;
  this->_info = 0;
}

template <typename T, typename M>
int ProjectorCgls<T, M>::Init() {
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  ASSERT(_A.IsInit());

  return 0;
}

template <typename T, typename M>
int ProjectorCgls<T, M>::Project(const T *x0, const T *y0, T s, T *x, T *y,
                                 T tol) {
  DEBUG_EXPECT(this->_done_init);
  DEBUG_EXPECT(s >= static_cast<T>(0.));
  if (!this->_done_init || s < static_cast<T>(0.))
    return 1;

  // Get Cublas handle
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  // Set initial x and y.
  cudaMemset(x, 0, _A.Cols() * sizeof(T));
  cudaMemcpy(y, y0, _A.Rows() * sizeof(T), cudaMemcpyDeviceToDevice);

  // y := y0 - Ax0;
  _A.Mul('n', static_cast<T>(-1.), x0, static_cast<T>(1.), y);

  // Minimize ||Ax - b||_2^2 + s||x||_2^2
  cgls::Solve(hdl, Gemv<T, M>(_A), static_cast<cgls::INT>(_A.Rows()),
      static_cast<cgls::INT>(_A.Cols()), y, x, s, tol, kMaxIter, kCglsQuiet);
  cudaDeviceSynchronize();
 
  // x := x + x0
  cml::vector<T> x_vec = cml::vector_view_array(x, _A.Cols());
  const cml::vector<T> x0_vec = cml::vector_view_array(x0, _A.Cols());
  cml::blas_axpy(hdl, static_cast<T>(1.), &x0_vec, &x_vec);
  cudaDeviceSynchronize();

  // y := Ax
  _A.Mul('n', static_cast<T>(1.), x, static_cast<T>(0.), y);
  cudaDeviceSynchronize();

#ifdef DEBUG
  // Verify that projection was successful.
  CheckProjection(&_A, x0, y0, x, y, s, static_cast<T>(1e1 * kTol));
#endif

  return 0;
}

template class ProjectorCgls<double, MatrixDense<double> >;
template class ProjectorCgls<float, MatrixDense<float> >;

template class ProjectorCgls<double, MatrixSparse<double> >;
template class ProjectorCgls<float, MatrixSparse<float> >;

}  // namespace pogs

