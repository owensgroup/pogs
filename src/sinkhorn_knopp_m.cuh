#ifndef SINKHORN_KNOPP_HPP_
#define SINKHORN_KNOPP_HPP_

#include "cml/cml_rand.cuh"

#include <mpi.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <limits>

#include "mpi_util.h"
#include "_interface_defs.h"
#include "cml/cml_blas.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_spblas.cuh"
#include "cml/cml_spmat.cuh"
#include "cml/cml_utils.cuh"
#include "cml/cml_vector.cuh"


namespace sinkhorn_knopp {

namespace {
__device__ inline double Abs(double x) { return fabs(x); }
__device__ inline float Abs(float x) { return fabsf(x); }
__device__ inline double Sqrt(double x) { return sqrt(x); }
__device__ inline float Sqrt(float x) { return sqrtf(x); }

// x -> |x|
template <typename T>
struct AbsF : thrust::unary_function<T, T> {
  __device__ T operator()(T x) { return fabs(x); }
};

// x -> 1 / x
template <typename T>
struct ReciprF : thrust::unary_function<T, T> {
  T alpha;
  ReciprF() : alpha(1) {}
  ReciprF(T alpha) : alpha(alpha) {}
  __device__ T operator()(T x) { return alpha / x; }
};

// x -> sqrt(x)
template <typename T>
struct SqrtF : thrust::unary_function<T, T> {
  __device__ T operator()(T x) { return Sqrt(x); }
};

// Sinkhorn Knopp algorithm for matrix equilibration.
// The following approx. holds: diag(d) * Ai * e =  1, diag(e) * Ai' * d = 1
// Output matrix is generated as: Ao = diag(d) * Ai * diag(e),
template <typename T>
void __global__ __SetSign(T* x, unsigned char *sign, size_t size) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x) {
    sign[t] = 0;
    for (unsigned int i = 0; i < 8; ++i) {
      sign[t] |= static_cast<unsigned char>(x[8 * t + i] < 0) << i; 
      x[8 * t + i] = x[8 * t + i] * x[8 * t + i];
    }
  }
}

template <typename T>
void __global__ __SetSignSingle(T* x, unsigned char *sign, size_t bits) {
  sign[0] = 0;
  for (unsigned int i = 0; i < bits; ++i) {
    sign[0] |= static_cast<unsigned char>(x[i] < 0) << i; 
    x[i] = x[i] * x[i];
  }
}

template <typename T>
void __global__ __UnSetSign(T* x, unsigned char *sign, size_t size) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x) {
    for (unsigned int i = 0; i < 8; ++i)
      x[8 * t + i] = (1 - 2 * static_cast<int>((sign[t] >> i) & 1)) * Sqrt(x[8 * t + i]);
  }
}

template <typename T>
void __global__ __UnSetSignSingle(T* x, unsigned char *sign, size_t bits) {
  for (unsigned int i = 0; i < bits; ++i)
    x[i] = (1 - 2 * static_cast<int>((sign[0] >> i) & 1)) * Sqrt(x[i]);
}


template <typename T, typename I, CBLAS_ORDER O>
T NormEst(cusparseHandle_t s_hdl, cublasHandle_t b_hdl,
          cusparseMatDescr_t descr, cml::spmat<T, I, O> *A_ij,
          int m, int n, int i_A) {
  const T kOne = static_cast<T>(1);
  const T kZero = static_cast<T>(0);
  const unsigned int kMaxIter = 50u;
  const T kTol = 1e-7;

  int kNodes;
  MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

  T nrm_est = 0, nrm_est_last;
  cml::vector<T> x = cml::vector_alloc<T>(n);
  cml::vector<T> Sx = cml::vector_alloc<T>(m);
  cml::rand(x.data, x.size);
  cudaDeviceSynchronize();

  cml::vector<T> Sx_sub = cml::vector_subvector(&Sx, i_A * A_ij->m, A_ij->m);

#ifndef POGS_OMPI_CUDA
  std::vector<T> x_h(x.size);
#else
  cml::matrix<T, CblasRowMajor> x_gather_buf =
    cml::matrix_calloc<T, CblasRowMajor>(kNodes, n);

  cml::vector<T> identity = cml::vector_alloc<T>(kNodes);
  cml::vector_set_all(&identity, kOne);
#endif

  unsigned int i = 0;
  for (/*unsigned int */i = 0; i < kMaxIter; ++i) {
    nrm_est_last = nrm_est;
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr,
        static_cast<T>(1), A_ij, &x, static_cast<T>(0), &Sx_sub);
    cudaDeviceSynchronize();
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_TRANSPOSE, descr,
        static_cast<T>(1), A_ij, &Sx_sub, static_cast<T>(0), &x);
    cudaDeviceSynchronize();

    // Reduce local matrix mults by summing partial column products together
#ifndef POGS_OMPI_CUDA
    cudaMemcpy(x_h.data(), x.data, x.size, cudaMemcpyDeviceToHost);
    mpiu::Allreduce(static_cast<T*>(MPI_IN_PLACE), x_h.data(), x_h.size(),
                    MPI_SUM, MPI_COMM_WORLD);
    cudaMemcpy(x.data, x_h.data(), x_h.size(), cudaMemcpyHostToDevice);
#else
    mpiu::Allgather(x.data, x.size, x_gather_buf.data, x.size, MPI_COMM_WORLD);
    cml::blas_gemv(b_hdl, CUBLAS_OP_T, kOne, &x_gather_buf, &identity,
                   kZero, &x);
#endif

    T nrmx = cml::blas_nrm2(b_hdl, &x);
    // Calculate global nrmSx by collecting partial values from all nodes
    T nrmSx = cml::blas_dot(b_hdl, &Sx, &Sx);
    mpiu::Allreduce(static_cast<T*>(MPI_IN_PLACE), &nrmSx, 1, MPI_SUM,
      MPI_COMM_WORLD);
    nrmSx = sqrtf(nrmSx);

    cml::vector_scale(&x, 1 / nrmx);
    nrm_est = nrmx / nrmSx;
    if (abs(nrm_est_last - nrm_est) < kTol * nrm_est)
      break;
  }
#ifdef POGS_OMPI_CUDA
  cml::matrix_free(&x_gather_buf);
  cml::vector_free(&identity);
#endif
  cml::vector_free(&x);
  cml::vector_free(&Sx);
  return nrm_est;
}

// Performs D * A * E for A in row major
template <typename T, typename I>
void __global__ __MultRow(T *d, T *e, T *val, I *row_ptr, I *col_ind,
                          size_t size) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x)
    for (unsigned int i = row_ptr[t]; i < row_ptr[t+1]; ++i)
      val[i] *= d[t] * e[col_ind[i]];
}

// Performs D * A * E for A in col major
template <typename T, typename I>
void __global__ __MultCol(T *d, T *e, T *val, I *col_ptr, I *row_ind,
                          size_t size) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x)
    for (unsigned int i = col_ptr[t]; i < col_ptr[t+1]; ++i)
      val[i] *= d[row_ind[i]] * e[t];
}

template <typename T, typename I>
void MultDiag(cml::vector<T> *d, cml::vector<T> *e,
              cml::spmat<T, I, CblasColMajor> *A) {
  int grid_dim_col = cml::calc_grid_dim(A->n, cml::kBlockSize);
  __MultCol<<<grid_dim_col, cml::kBlockSize>>>(d->data, e->data, A->val,
      A->ptr, A->ind, A->n);
  int grid_dim_row = cml::calc_grid_dim(A->m, cml::kBlockSize);
  __MultRow<<<grid_dim_row, cml::kBlockSize>>>(d->data, e->data,
      A->val + A->nnz, A->ptr + cml::ptr_len(*A), A->ind + A->nnz, A->m);
}

template <typename T, typename I>
void MultDiag(cml::vector<T> *d, cml::vector<T> *e,
              cml::spmat<T, I, CblasRowMajor> *A) {
  int grid_dim_row = cml::calc_grid_dim(A->m, cml::kBlockSize);
  __MultRow<<<grid_dim_row, cml::kBlockSize>>>(d->data, e->data, A->val,
      A->ptr, A->ind, A->m);
  int grid_dim_col = cml::calc_grid_dim(A->n, cml::kBlockSize);
  __MultCol<<<grid_dim_col, cml::kBlockSize>>>(d->data, e->data,
      A->val + A->nnz, A->ptr + cml::ptr_len(*A), A->ind + A->nnz, A->n);
}

}  // namespace

template <typename T, typename I, CBLAS_ORDER O>
int Equilibrate(cusparseHandle_t s_hdl, cublasHandle_t b_hdl,
                cusparseMatDescr_t descr, cml::spmat<T, I, O> *A_ij,
                cml::vector<T> *d, cml::vector<T> *e,
                int m, int n, int i_A) {
  const T kOne = static_cast<T>(1);
  const T kZero = static_cast<T>(0);

  int kNodes;
  MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

  cml::vector_set_all(d, static_cast<T>(1));
  cml::vector_set_all(e, static_cast<T>(1));

  cml::vector<T> d_sub = cml::vector_subvector(d, i_A * A_ij->m, A_ij->m); 

#ifndef POGS_OMPI_CUDA
  std::vector<T> e_h{e->size};
  std::vector<T> d_h{d->size};
#else
  cml::matrix<T, CblasRowMajor> e_gather_buf =
    cml::matrix_calloc<T, CblasRowMajor>(kNodes, n);

  cml::vector<T> identity = cml::vector_alloc<T>(kNodes);
  cml::vector_set_all(&identity, kOne);
#endif
//  return 0;

  unsigned int kNumItr = 10;
  // Create bit-vector with sign.
  unsigned char *sign;
  size_t num_sign_bytes = (2 * A_ij->nnz + 7) / 8;
  cudaError_t err = cudaMalloc(&sign, num_sign_bytes);
  CudaCheckError(err);
  if (err != cudaSuccess)
    return 1;

  int num_chars = (2 * A_ij->nnz) / 8;
  int grid_size = cml::calc_grid_dim(num_chars, cml::kBlockSize);
  __SetSign<<<grid_size, cml::kBlockSize>>>(A_ij->val, sign, num_chars);
  cudaDeviceSynchronize();
  if (2 * A_ij->nnz > num_chars * 8) {
    __SetSignSingle<<<1, 1>>>(A_ij->val + num_chars * 8, sign + num_chars, 
        2 * A_ij->nnz - num_chars * 8);
    cudaDeviceSynchronize();
  }

  for (unsigned int k = 0; k < kNumItr; ++k) {
    // Perform local matrix mult of portion of d onto e
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_TRANSPOSE, descr,
        static_cast<T>(1), A_ij, &d_sub, static_cast<T>(0), e);
    cudaDeviceSynchronize();

    // Reduce local matrix mults by summing partial column products together
    // Must sum because we are multiplying by the transpose 
#ifndef POGS_OMPI_CUDA
    cudaMemcpy(e_h.data(), e->data, e->size, cudaMemcpyDeviceToHost);
    mpiu::Allreduce(static_cast<T*>(MPI_IN_PLACE), e_h.data(), e_h.size(),
                    MPI_SUM, MPI_COMM_WORLD);
    cudaMemcpy(e->data, e_h.data(), e_h.size(), cudaMemcpyHostToDevice);
#else
    mpiu::Allgather(e->data, e->size, e_gather_buf.data, e->size,
                    MPI_COMM_WORLD);
    cml::blas_gemv(b_hdl, CUBLAS_OP_T, kOne, &e_gather_buf, &identity,
                   kZero, e);
#endif
    cml::vector_add_constant(e, static_cast<T>(1e-4));
    cudaDeviceSynchronize();
    thrust::transform(thrust::device_pointer_cast(e->data),
        thrust::device_pointer_cast(e->data + e->size),
        thrust::device_pointer_cast(e->data), ReciprF<T>(m));
    cudaDeviceSynchronize();
    

    // Perform local matrix mult of portion of e onto d
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr,
        static_cast<T>(1), A_ij, e, static_cast<T>(0), &d_sub);
    cudaDeviceSynchronize();
    cml::vector_add_constant(&d_sub, static_cast<T>(1e-4));
    cudaDeviceSynchronize();
    thrust::transform(thrust::device_pointer_cast(d_sub.data),
        thrust::device_pointer_cast(d_sub.data + d_sub.size),
        thrust::device_pointer_cast(d_sub.data), ReciprF<T>(n));
    cudaDeviceSynchronize();
  }
//  return 0;

  thrust::transform(thrust::device_pointer_cast(d_sub.data),
      thrust::device_pointer_cast(d_sub.data + d_sub.size),
      thrust::device_pointer_cast(d_sub.data), SqrtF<T>());
  cudaDeviceSynchronize();

  // Gather d on each node
#ifndef POGS_OMPI_CUDA
  cudaMemcpy(d_h.data(), d->data, d->size, cudaMemcpyDeviceToHost);
  mpiu::Allgather(static_cast<T*>(MPI_IN_PLACE), 0, d_h.data(), d_sub.size,
      MPI_COMM_WORLD);
  cudaMemcpy(d.data, d_h.data(), d_h.size(), cudaMemcpyHostToDevice);
#else
  mpiu::Allgather(static_cast<T*>(MPI_IN_PLACE), 0, d->data, d_sub.size,
      MPI_COMM_WORLD);
#endif

  thrust::transform(thrust::device_pointer_cast(e->data),
      thrust::device_pointer_cast(e->data + e->size),
      thrust::device_pointer_cast(e->data), SqrtF<T>());
  cudaDeviceSynchronize();

  __UnSetSign<<<grid_size, cml::kBlockSize>>>(A_ij->val, sign, num_chars);
  cudaDeviceSynchronize();
  if (2 * A_ij->nnz > num_chars * 8) {
    __UnSetSignSingle<<<1, 1>>>(A_ij->val + num_chars * 8, sign + num_chars, 
        2 * A_ij->nnz - num_chars * 8);
    cudaDeviceSynchronize();
  }
  
  // Compute D * A * E

  // Choose portions of d and e that line up with local matrix block
  MultDiag(&d_sub, e, A_ij);
  cudaDeviceSynchronize();

  T nrmA = NormEst(s_hdl, b_hdl, descr, A_ij, m, n, i_A);
  T nrmd = cml::blas_nrm2(b_hdl, d);
  T nrme = cml::blas_nrm2(b_hdl, e);
  T scale = sqrt(nrmd * sqrt(e->size) / (nrme * sqrt(d->size)));

  cml::vector<T> a_vec = cml::vector_view_array(A_ij->val, 2 * A_ij->nnz);
  cml::vector_scale(&a_vec, 1 / nrmA);
  cudaDeviceSynchronize();

  cml::vector_scale(d, 1 / (scale * sqrt(nrmA)));
  cudaDeviceSynchronize();
  cml::vector_scale(e, scale / sqrt(nrmA));
  cudaDeviceSynchronize();

  cudaFree(sign);

#ifdef POGS_OMPI_CUDA
  cml::matrix_free(&e_gather_buf);  
  cml::vector_free(&identity);
#endif

  return 0;
}

// template <typename T>
// void SinkhornKnopp(cublasHandle_t handle, const T *Ai, cml::matrix<T> *Ao,
//                    cml::vector<T> *d, cml::vector<T> *e) {
//   unsigned int kNumItr = 10;
//   cml::matrix_memcpy(Ao, Ai);
//   cml::vector_set_all(d, static_cast<T>(1));
// 
//   // A := |A| -- elementwise
//   thrust::transform(thrust::device_pointer_cast(Ao->data),
//       thrust::device_pointer_cast(Ao->data + Ao->size2 * Ao->tda),
//       thrust::device_pointer_cast(Ao->data), AbsF<T>());
// 
//   // e := 1 ./ A' * d; d := 1 ./ A * e; -- k times.
//   for (unsigned int k = 0; k < kNumItr; ++k) {
//     cml::blas_gemv(handle, CUBLAS_OP_T, static_cast<T>(1), Ao, d,
//         static_cast<T>(0), e);
// 
//     thrust::transform(thrust::device_pointer_cast(e->data),
//         thrust::device_pointer_cast(e->data + e->size),
//         thrust::device_pointer_cast(e->data), ReciprF<T>());
// 
//     cml::blas_gemv(handle, CUBLAS_OP_N, static_cast<T>(1), Ao, e,
//         static_cast<T>(0), d);
//     thrust::transform(thrust::device_pointer_cast(d->data),
//         thrust::device_pointer_cast(d->data + d->size),
//         thrust::device_pointer_cast(d->data), ReciprF<T>());
// 
//     T nrm_d = cml::blas_nrm2(handle, d) / sqrt(static_cast<T>(d->size));
//     T nrm_e = cml::blas_nrm2(handle, e) / sqrt(static_cast<T>(e->size));
//     T scale = sqrt(nrm_e / nrm_d);
//     cml::blas_scal(handle, scale, d);
//     cml::blas_scal(handle, static_cast<T>(1) / scale, e);
//   }
// 
//   // A := D * A * E
//   T* de = new T[std::max(Ao->size1, Ao->size2)];
//   cml::vector_memcpy(de, d);
//   cml::matrix_memcpy(Ao, Ai);
//   for (unsigned int i = 0; i < Ao->size1; ++i) {
//     cml::vector<T> v = cml::matrix_row(Ao, i);
//     cml::blas_scal(handle, de[i], &v);
//   }
//   cml::vector_memcpy(de, e);
//   for (unsigned int j = 0; j < Ao->size2; ++j) {
//     cml::vector<T> v = cml::matrix_column(Ao, j);
//     cml::blas_scal(handle, de[j], &v);
//   }
//   delete [] de;
// }

// template <typename T, CBLAS_ORDER O>
// int Equilibrate(T *A, cml::vector<T> *d, cml::vector<T> *e) {
//   int err = 0;
//   T *dpr = new T[d->size], *epr = new T[e->size];
//   if (d->size < e->size) {
//     for (unsigned int i = 0; i < d->size; ++i)
//       dpr[i] = 0;
//     for (unsigned int j = 0; j < e->size; ++j)
//       epr[j] = 1;
//     if (O == CblasColMajor)
//       for (unsigned int j = 0; j < e->size; ++j)
// #pragma omp parallel for
//         for (unsigned int i = 0; i < d->size; ++i)
//           dpr[i] += std::fabs(A[i + j * d->size]);
//     else
// #pragma omp parallel for
//       for (unsigned int i = 0; i < d->size; ++i)
//         for (unsigned int j = 0; j < e->size; ++j)
//           dpr[i] += std::fabs(A[i * e->size + j]);
//     for (unsigned int i = 0; i < d->size; ++i) {
//       err += dpr[i] == 0;
//       dpr[i] = 1 / dpr[i];
//     }
//   } else {
//     for (unsigned int i = 0; i < d->size; ++i)
//       dpr[i] = 1;
//     for (unsigned int j = 0; j < e->size; ++j)
//       epr[j] = 0;
//     if (O == CblasColMajor)
// #pragma omp parallel for
//       for (unsigned int j = 0; j < e->size; ++j)
//         for (unsigned int i = 0; i < d->size; ++i)
//           epr[j] += std::fabs(A[i + j * d->size]);
//     else
//       for (unsigned int i = 0; i < d->size; ++i)
// #pragma omp parallel for
//         for (unsigned int j = 0; j < e->size; ++j)
//           epr[j] += std::fabs(A[i * e->size + j]);
//     for (unsigned int j = 0; j < e->size; ++j) {
//       err += epr[j] == 0;
//       epr[j] = 1 / epr[j];
//     }
//   }
//   if (O == CblasColMajor)
// #pragma omp parallel for
//     for (unsigned int j = 0; j < e->size; ++j)
//       for (unsigned int i = 0; i < d->size; ++i)
//         A[i + j * d->size] *= epr[j] * dpr[i];
//   else
// #pragma omp parallel for
//     for (unsigned int i = 0; i < d->size; ++i)
//       for (unsigned int j = 0; j < e->size; ++j)
//         A[i * e->size + j] *= epr[j] * dpr[i];
//   cml::vector_memcpy(d, dpr);
//   cml::vector_memcpy(e, epr);
//   delete [] dpr;
//   delete [] epr;
//   if (err)
//     Printf("Error: Zero column/row in A\n");
//   return err;
// }

//*************************************************************************


template <typename T>
void __global__ __recipr(T* x, size_t size, size_t stride) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x)
    x[t * stride] = 1 / x[t * stride];
}

template <typename T>
void __global__ __equi(T* A, size_t m, size_t n, size_t tda, T *x,
                       size_t incx) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int j = tid; j < n; j += gridDim.x * blockDim.x) {
    x[j * incx] = 0;
    for (unsigned int i = 0; i < m; ++i)
      x[j * incx] += A[i * tda + j] * A[i * tda + j];
    x[j * incx] = 1 / sqrt(x[j * incx]);
  }
}

template <typename T, CBLAS_ORDER O, CBLAS_SIDE S>
void __global__ __diag_mult(T* A, size_t m, size_t n, size_t tda, T *x,
                            size_t incx) {
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  if (O == CblasRowMajor) {
    if (S == CblasRight) {
      for (unsigned int i = tidx; i < m; i += gridDim.x * blockDim.x) {
        for (unsigned int j = tidy; j < n; j += gridDim.y * blockDim.y) {
          A[i * tda + j] *= x[j * incx];
        }
      }
    } else {
      for (unsigned int i = tidx; i < m; i += gridDim.x * blockDim.x) {
        for (unsigned int j = tidy; j < n; j += gridDim.y * blockDim.y) {
          A[i * tda + j] *= x[i * incx];
        }
      }
    }
  } else {
    if (S == CblasRight) {
      for (unsigned int j = tidy; j < n; j += gridDim.y * blockDim.y) {
        for (unsigned int i = tidx; i < m; i += gridDim.x * blockDim.x) {
          A[i + j * tda] *= x[j * incx];
        }
      }
    } else {
      for (unsigned int j = tidy; j < n; j += gridDim.y * blockDim.y) {
        for (unsigned int i = tidx; i < m; i += gridDim.x * blockDim.x) {
          A[i + j * tda] *= x[i * incx];
        }
      }
    }
  }
}

template <typename T>
void __global__ __any_is_zero(T *x, size_t size, size_t incx, int *tf) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < size; i += gridDim.x * blockDim.x)
    if (x[i * incx] == static_cast<T>(0))
      tf[0] = 1;
}


template <typename T, CBLAS_ORDER O>
void EquiOp(cml::matrix<T, O> *A, cml::vector<T> *x) {
  unsigned int grid_dim = cml::calc_grid_dim(x->size, cml::kBlockSize);
  size_t m;
  if (O == CblasRowMajor)
    m = A->size1;
  else
    m = A->size2;
  __equi<T><<<grid_dim, cml::kBlockSize>>>(A->data, m, x->size, A->tda, x->data,
      x->stride);
}

template <typename T>
void ReciprOp(cml::vector<T> *x) {
  unsigned int grid_dim = cml::calc_grid_dim(x->size, cml::kBlockSize);
  __recipr<<<grid_dim, cml::kBlockSize>>>(x->data, x->size, x->stride);
}

template <typename T, CBLAS_ORDER O, CBLAS_SIDE S>
void DiagOp(cml::matrix<T, O> *A, cml::vector<T> *x) {
  const size_t kBlockSize = std::sqrt<size_t>(cml::kBlockSize);
  dim3 grid_size, block_size;
  grid_size.x = cml::calc_grid_dim(A->size1, kBlockSize);
  grid_size.y = cml::calc_grid_dim(A->size2, kBlockSize); 
  block_size.x = kBlockSize;
  block_size.y = kBlockSize;
  __diag_mult<T, O, S><<<grid_size, block_size>>>(A->data, A->size1, A->size2,
      A->tda, x->data, x->stride);
}

template <typename T>
int AnyIsZeroOp(cml::vector<T> *x) {
  int tf_h = 0, *tf_d;
  cudaMalloc(&tf_d, sizeof(int));
  cudaMemcpy(tf_d, &tf_h, sizeof(int), cudaMemcpyHostToDevice);
  unsigned int grid_dim = cml::calc_grid_dim(x->size, cml::kBlockSize);
  __any_is_zero<<<grid_dim, cml::kBlockSize>>>(x->data, x->size, x->stride,
      tf_d);
  cudaMemcpy(&tf_h, tf_d, sizeof(int), cudaMemcpyDeviceToHost);
  return tf_h;
}

template <typename T>
int Equilibrate(cml::matrix<T, CblasRowMajor> *A,
                cml::vector<T> *d, cml::vector<T> *e) {
  int err;
  if (A->size1 < A->size2) {
    unsigned int num_h = 4;
    cublasHandle_t h[num_h];
    for (unsigned int i = 0; i < num_h; ++i)
      cublasCreate(h + i);
    T *dpr = new T[A->size1];
    cml::vector_set_all(e, static_cast<T>(1));
    for (unsigned int i = 0; i < A->size1; ++i) {
      cml::vector<T> v = cml::matrix_row(A, i);
      dpr[i] = cml::blas_nrm2(h[i % num_h], &v);
    }
    cml::vector_memcpy(d, dpr);
    ReciprOp(d);
    DiagOp<T, CblasRowMajor, CblasLeft>(A, d);
    delete [] dpr;
    for (unsigned int i = 0; i < num_h; ++i)
      cublasDestroy(h[i]);
    err = AnyIsZeroOp(d);
  } else {
    cml::vector_set_all(d, static_cast<T>(1));
    EquiOp(A, e);
    DiagOp<T, CblasRowMajor, CblasRight>(A, e);
    err = AnyIsZeroOp(e);
  }
  return err;
}

template <typename T>
int Equilibrate(cml::matrix<T, CblasColMajor> *A,
                cml::vector<T> *d, cml::vector<T> *e) {
  int err;
  if (A->size1 < A->size2) {
    cml::vector_set_all(e, static_cast<T>(1));
    EquiOp(A, d);
    DiagOp<T, CblasColMajor, CblasLeft>(A, d);
    err = AnyIsZeroOp(d);
  } else {
    unsigned int num_h = 4;
    cublasHandle_t h[num_h];
    for (unsigned int i = 0; i < num_h; ++i)
      cublasCreate(h + i);
    T *epr = new T[A->size2];
    cml::vector_set_all(d, static_cast<T>(1));
    for (unsigned int j = 0; j < A->size2; ++j) {
      cml::vector<T> v = cml::matrix_column(A, j);
      epr[j] = cml::blas_nrm2(h[j % num_h], &v);
    }
    cml::vector_memcpy(e, epr);
    ReciprOp(e);
    DiagOp<T, CblasColMajor, CblasRight>(A, e);
    delete [] epr;
    for (unsigned int i = 0; i < num_h; ++i)
      cublasDestroy(h[i]);
    err = AnyIsZeroOp(e);
  }
  return err;
}

}  // namespace sinkhorn_knopp

#endif  // SINKHORN_KNOPP_HPP_

