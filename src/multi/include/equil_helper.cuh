#ifndef EQUIL_HELPER_CUH_
#define EQUIL_HELPER_CUH_

#include <thrust/functional.h>

#include "cml/cml_blas.cuh"
#include "cml/cml_rand.cuh"
#include "cml/cml_vector.cuh"
#include "matrix/matrix_dist.h"
#include "util.h"
#include "mpi_helper.h"

namespace pogs {
namespace {

template<class T, class U>
struct is_same {
  enum { value = 0 };
};

template<class T>
struct is_same<T, T> {
  enum { value = 1 };
};

// Different norm types.
enum NormTypes { kNorm1, kNorm2, kNormFro };

// TODO: Figure out a better value for this constant
const double kSinkhornConst        = 1e-8;
const double kNormEstTol           = 1e-3;
const unsigned int kEquilIter      = 50u;
const unsigned int kNormEstMaxIter = 50u;

////////////////////////////////////////////////////////////////////////////////
///////////////////////// Helper Functions /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct ReciprF : thrust::unary_function<T, T> {
  T alpha;
  __host__ __device__ ReciprF() : alpha(1) { }
  __host__ __device__ ReciprF(T alpha) : alpha(alpha) { }
  __host__ __device__ T operator()(T x) { return alpha / x; }
};

template <typename T>
struct AbsF : thrust::unary_function<T, T> {
  __host__ __device__ inline double Abs(double x) { return fabs(x); }
  __host__ __device__ inline float Abs(float x) { return fabsf(x); }
  __host__ __device__ T operator()(T x) { return Abs(x); }
};

template <typename T>
struct IdentityF: thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return x; }
};

template <typename T>
struct SquareF: thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T x) { return x * x; }
};

template <typename T>
struct SqrtF : thrust::unary_function<T, T> {
  __host__ __device__ inline double Sqrt(double x) { return sqrt(x); }
  __host__ __device__ inline float Sqrt(float x) { return sqrtf(x); }
  __host__ __device__ T operator()(T x) { return Sqrt(x); }
};

template <typename T, typename F>
void __global__ __SetSign(T* x, unsigned char *sign, size_t size, F f) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x) {
    sign[t] = 0;
    for (unsigned int i = 0; i < 8; ++i) {
      sign[t] |= static_cast<unsigned char>(x[8 * t + i] < 0) << i; 
      x[8 * t + i] = f(x[8 * t + i]);
    }
  }
}

template <typename T, typename F>
void __global__ __SetSignSingle(T* x, unsigned char *sign, size_t bits, F f) {
  sign[0] = 0;
  for (unsigned int i = 0; i < bits; ++i) {
    sign[0] |= static_cast<unsigned char>(x[i] < 0) << i; 
    x[i] = f(x[i]);
  }
}

template <typename T, typename F>
void __global__ __UnSetSign(T* x, unsigned char *sign, size_t size, F f) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int t = tid; t < size; t += gridDim.x * blockDim.x) {
    for (unsigned int i = 0; i < 8; ++i) {
      x[8 * t + i] = (1 - 2 * static_cast<int>((sign[t] >> i) & 1)) *
          f(x[8 * t + i]);
    }
  }
}

template <typename T, typename F>
void __global__ __UnSetSignSingle(T* x, unsigned char *sign, size_t bits, F f) {
  for (unsigned int i = 0; i < bits; ++i)
    x[i] = (1 - 2 * static_cast<int>((sign[0] >> i) & 1)) * f(x[i]);
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////// Norm Estimation //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
T Norm2Est(cublasHandle_t hdl, const MatrixDist<T> *A) {
  // Same as MATLAB's method for norm estimation.

  T kTol = static_cast<T>(kNormEstTol);

  MPI_Datatype t_type = (is_same<T,double>::value ?
                         MPI_DOUBLE :
                         MPI_FLOAT);

  const BlockMeta &block = A->Meta().block;
  MPI_Comm x_comm, Sx_comm;
  MPI_Comm_split(MPI_COMM_WORLD, block.column, 0, &x_comm);
  MPI_Comm_split(MPI_COMM_WORLD, block.row, 0, &Sx_comm);

  T norm_est = 0, norm_est_last;
  cml::vector<T> x = cml::vector_alloc<T>(block.Cols());
  cml::vector<T> Sx = cml::vector_alloc<T>(block.Rows());

  cml::vector<T> x_temp = cml::vector_alloc<T>(block.Cols());
  cml::vector<T> Sx_temp = cml::vector_alloc<T>(block.Rows());

  cml::rand(x.data, x.size);
  cudaDeviceSynchronize();

  unsigned int i = 0;
  for (i = 0; i < kNormEstMaxIter; ++i) {
    norm_est_last = norm_est;
    A->BlockMul('n', static_cast<T>(1.), x.data, static_cast<T>(0.), Sx.data);
    cudaDeviceSynchronize();
    // Reduce local matrix mults by summing partial column products together
    mpih::Allreduce(hdl, Sx.data, Sx_temp.data, Sx.size, MPI_SUM, Sx_comm);
    cudaDeviceSynchronize();
    cml::vector_memcpy(&Sx, &Sx_temp);

    A->BlockMul('t', static_cast<T>(1.), Sx.data, static_cast<T>(0.), x.data);
    cudaDeviceSynchronize();
    // Reduce local matrix mults by summing partial column products together
    mpih::Allreduce(hdl, x.data, x_temp.data, x.size, MPI_SUM, x_comm);
    cudaDeviceSynchronize();
    cml::vector_memcpy(&x, &x_temp);

    // Calculate global norms by collecting partial row norm values
    T normx;
    cml::blas_dot(hdl, &x, &x, &normx);
    cudaDeviceSynchronize();
    T tempx;
    MPI_Allreduce(&normx, &tempx, 1, t_type, MPI_SUM, Sx_comm);
    cudaDeviceSynchronize();
    normx = tempx;
    normx = sqrtf(normx);
    T normSx;
    cml::blas_dot(hdl, &Sx, &Sx, &normSx);
    cudaDeviceSynchronize();
    T tempSx;
    MPI_Allreduce(&normSx, &tempSx, 1, t_type, MPI_SUM, x_comm);
    cudaDeviceSynchronize();
    normSx = tempSx;
    normSx = sqrtf(normSx);

    cml::vector_scale(&x, 1 / normx);
    norm_est = normx / normSx;
    if (abs(norm_est_last - norm_est) < kTol * norm_est)
      break;
  }
  DEBUG_EXPECT_LT(i, kNormEstMaxIter);

  cml::vector_free(&x);
  cml::vector_free(&Sx);

  MPI_Comm_free(&Sx_comm);
  MPI_Comm_free(&x_comm);
  return norm_est;
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////// Modified Sinkhorn Knopp //////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
void SinkhornKnopp(cublasHandle_t hdl, const MatrixDist<T> *A, T *d, T *e) {
  MPI_Datatype t_type = (is_same<T,double>::value ?
                         MPI_DOUBLE :
                         MPI_FLOAT);
  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  const BlockMeta &block = A->Meta().block;

  printf("%d before comm splits\n", kRank);
  MPI_Comm row_comm, col_comm;
  //MPI_Comm_split(MPI_COMM_WORLD, block.row, 0, &row_comm);
  MPI_Comm_dup(MPI_COMM_SELF, &row_comm);
  printf("%d after row splits\n", kRank);
  //MPI_Comm_split(MPI_COMM_WORLD, block.column, 0, &col_comm);
  MPI_Comm_dup(MPI_COMM_WORLD, &col_comm);
  printf("%d before col splits\n", kRank);
  int row_size, col_size;
  MPI_Comm_size(row_comm, &row_size);
  MPI_Comm_size(col_comm, &col_size);

  printf("%d after comm splits\n", kRank);
  cml::vector<T> d_vec = cml::vector_view_array<T>(d, block.Rows());
  cml::vector<T> e_vec = cml::vector_view_array<T>(e, block.Cols());
  cml::vector_set_all(&d_vec, static_cast<T>(1.));
  cml::vector_set_all(&e_vec, static_cast<T>(1.));

  cml::vector<T> d_vec_temp = cml::vector_calloc<T>(block.Rows());
  cml::vector<T> e_vec_temp = cml::vector_calloc<T>(block.Cols());

  printf("%d before sinkhorn iterate\n", kRank);
  for (unsigned int k = 0; k < kEquilIter; ++k) {
    // e := 1 ./ (A' * d).
    cudaDeviceSynchronize();
    A->BlockMul('t', static_cast<T>(1.), d, static_cast<T>(0.), e);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
    if (col_size > 1) {
      mpih::Allreduce(hdl, e_vec.data, e_vec_temp.data, e_vec.size, MPI_SUM,
                      col_comm);
      cudaDeviceSynchronize();
      cml::vector_memcpy(&e_vec, &e_vec_temp);
    }

    cml::vector_add_constant(&e_vec,
        static_cast<T>(kSinkhornConst) * (A->Rows() + A->Cols()) / A->Rows());
    cudaDeviceSynchronize();
    thrust::transform(thrust::device_pointer_cast(e),
        thrust::device_pointer_cast(e + e_vec.size),
        thrust::device_pointer_cast(e), ReciprF<T>(A->Rows()));
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    // d := 1 ./ (A' * e).
    A->BlockMul('n', static_cast<T>(1.), e, static_cast<T>(0.), d);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    // Must check size here because when RDMA is occuring the master node
    // suffers a no op on this???
    if (row_size > 1) {
      mpih::Allreduce(hdl, d_vec.data, d_vec_temp.data, d_vec.size, MPI_SUM,
                      row_comm);
      cudaDeviceSynchronize();
      cml::vector_memcpy(&d_vec, &d_vec_temp);
    }

    cml::vector_add_constant(&d_vec,
        static_cast<T>(kSinkhornConst) * (A->Rows() + A->Cols()) / A->Cols());
    cudaDeviceSynchronize();
    thrust::transform(thrust::device_pointer_cast(d),
        thrust::device_pointer_cast(d + d_vec.size),
        thrust::device_pointer_cast(d), ReciprF<T>(A->Cols()));
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
}

}  // namespace
}  // namespace pogs

#endif  // EQUIL_HELPER_CUH_
