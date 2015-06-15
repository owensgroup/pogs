#include <cublas_v2.h>

#include "cml/cml_blas.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_vector.cuh"
#include "equil_helper.cuh"
#include "matrix/matrix_dist.h"
#include "matrix/matrix_dist_dense.h"
#include "util.h"
#include "mpi_helper.h"
#include "timer.h"


namespace pogs {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helper Functions ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// File scoped constants.
const NormTypes kNormEquilibrate = kNorm2;
const NormTypes kNormNormalize   = kNormFro;

template<typename T>
struct GpuData {
  const T *orig_data;
  cublasHandle_t handle;
  GpuData(const T *orig_data) : orig_data(orig_data) {
    cublasCreate(&handle);
    DEBUG_CUDA_CHECK_ERR();
  }
  ~GpuData() {
    cublasDestroy(handle);
    DEBUG_CUDA_CHECK_ERR();
  }
};

cublasOperation_t OpToCublasOp(char trans) {
  ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return trans == 'n' || trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
}

template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type, const MatrixDistDense<T>& A);

template <typename T>
void MultDiag(const T *d, const T *e, size_t m, size_t n,
              typename MatrixDistDense<T>::Ord ord, T *data);

template <typename T>
void DistributeBlocks(const Schedule &s, typename MatrixDistDense<T>::Ord ord,
                      const size_t m, const size_t n,
                      const T *orig_data, T *gpu_data);

}  // namespace

////////////////////////////////////////////////////////////////////////////////
/////////////////////// MatrixDistDense Implementation /////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
MatrixDistDense<T>::MatrixDistDense(const Schedule &s, char ord,
                                    size_t m, size_t n, const T *data)
  : MatrixDist<T>(s, m, n), _data(0) {
  ASSERT(ord == 'r' || ord == 'R' || ord == 'c' || ord == 'C');
  _ord = (ord == 'r' || ord == 'R') ? ROW : COL;

  cudaSetDevice(MatrixDist<T>::_meta.gpu_indicies[0]);

  // Set GPU specific _info.
  GpuData<T> *info = new GpuData<T>(data);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixDistDense<T>::MatrixDistDense(const MatrixDistDense<T>& A)
  : MatrixDist<T>(A._S, A._m, A._n), _data(0), _ord(A._ord) {

  GpuData<T> *info_A = reinterpret_cast<GpuData<T>*>(A._info);
  GpuData<T> *info = new GpuData<T>(info_A->orig_data);
  this->_info = reinterpret_cast<void*>(info);
}

template <typename T>
MatrixDistDense<T>::~MatrixDistDense() {
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  delete info;
  this->_info = 0;

  if (this->_done_init && _data) {
    cudaFree(_data);
    this->_data = 0;
    DEBUG_CUDA_CHECK_ERR();
  }
}

template <typename T>
int MatrixDistDense<T>::Init() {
  DEBUG_EXPECT(!this->_done_init);
  if (this->_done_init)
    return 1;
  this->_done_init = true;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);

  // Distribute Matrix and copy to GPU.
  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  const BlockMeta &block = MatrixDist<T>::_S.At(kRank).block;
  size_t rows = block.Rows();
  size_t columns = block.Cols();

  double mat_malloc = timer<double>();
  cudaMalloc(&_data, rows * columns * sizeof(T));
  mat_malloc = timer<double>() - mat_malloc;

  double dist_mat = timer<double>();
  DistributeBlocks(MatrixDist<T>::_S,
                   _ord,
                   MatrixDist<T>::_m,
                   MatrixDist<T>::_n,
                   info->orig_data,
                   _data);
  DEBUG_CUDA_CHECK_ERR();
  dist_mat = timer<double>() - dist_mat;

  MASTER(kRank) {
    BMARK_PRINT_T("matrix_malloc_time", mat_malloc);
    BMARK_PRINT_T("distribute_matrix_time", dist_mat);
  }
  return 0;
}

template <typename T>
int MatrixDistDense<T>::BlockMul(char trans, T alpha, const T *x, T beta, T *y)
  const {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init)
    return 1;

  const BlockMeta &block = MatrixDist<T>::_meta.block;

  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  const cml::vector<T> x_vec = cml::vector_view_array<T>(x, block.Cols());
  cml::vector<T> y_vec = cml::vector_view_array<T>(y, block.Rows());

  if (_ord == ROW) {
    cml::matrix<T, CblasRowMajor> A =
      cml::matrix_view_array<T, CblasRowMajor>(_data, block.Rows(),
                                               block.Cols());
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  } else {
    cml::matrix<T, CblasColMajor> A =
      cml::matrix_view_array<T, CblasColMajor>(_data, block.Rows(),
                                               block.Cols());
    cml::blas_gemv(hdl, OpToCublasOp(trans), alpha, &A, &x_vec, beta, &y_vec);
  }
  CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
int MatrixDistDense<T>::Mul(char trans, T alpha, const T *x, T beta, T *y)
  const {
  DEBUG_EXPECT(this->_done_init);
  if (!this->_done_init)
    return 1;

  MPI_Datatype t_type = (is_same<T,double>::value ?
                         MPI_DOUBLE :
                         MPI_FLOAT);

  const BlockMeta &block = MatrixDist<T>::_meta.block;

  MPI_Comm row_comm, col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, block.row, 0, &row_comm);
  MPI_Comm_split(MPI_COMM_WORLD, block.column, 0, &col_comm);

  const cml::vector<T> x_vec = cml::vector_view_array<T>(x, this->_n);
  cml::vector<T> y_vec = cml::vector_view_array<T>(y, this->_m);

  const cml::vector<T> x_vec_block =
    cml::vector_view_array<T>(x + block.column_begin, block.Cols());
  cml::vector<T> y_vec_block =
    cml::vector_view_array<T>(y + block.row_begin, block.Rows());

  BlockMul(trans, alpha, x_vec_block.data, beta, y_vec_block.data);
  if (OpToCublasOp(trans) == CUBLAS_OP_N) {
    // Sum partial components
    MPI_Allreduce(MPI_IN_PLACE, y_vec_block.data, y_vec_block.size, t_type,
                  MPI_SUM, col_comm);
    // Compose components
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  y_vec.data, y_vec.size, t_type,
                  row_comm);
  } else {
    // Sum partial components
    MPI_Allreduce(MPI_IN_PLACE, y_vec_block.data, y_vec_block.size, t_type,
                  MPI_SUM, row_comm);
    // Compose components
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  y_vec.data, y_vec.size, t_type,
                  col_comm);
  }

  CUDA_CHECK_ERR();

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  return 0;
}

template <typename T>
int MatrixDistDense<T>::Equil(T *d, T *e) {
  DEBUG_ASSERT(this->_done_init);
  if (!this->_done_init)
    return 1;

  double t0 = timer<double>();

  // Extract cublas handle from _info.
  GpuData<T> *info = reinterpret_cast<GpuData<T>*>(this->_info);
  cublasHandle_t hdl = info->handle;

  const BlockMeta &block = MatrixDist<T>::_meta.block;

  // Number of elements in local matrix.
  size_t num_el = block.Rows() * block.Cols();

  // Create bit-vector with signs of entries in A and then let A = f(A),
  // where f = |A| or f = |A|.^2.
  unsigned char *sign;
  size_t num_sign_bytes = (num_el + 7) / 8;
  cudaMalloc(&sign, num_sign_bytes);
  CUDA_CHECK_ERR();

  // Fill sign bits, assigning each thread a multiple of 8 elements.
  size_t num_chars = num_el / 8;
  size_t grid_size = cml::calc_grid_dim(num_chars, cml::kBlockSize);
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    __SetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        SquareF<T>());
  } else {
    __SetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        AbsF<T>());
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // If numel(A) is not a multiple of 8, then we need to set the last couple
  // of sign bits too.
  if (num_el > num_chars * 8) {
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
      __SetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars,
          num_el - num_chars * 8, SquareF<T>());
    } else {
      __SetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars,
          num_el - num_chars * 8, AbsF<T>());
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  // Perform Sinkhorn-Knopp equilibration.
  SinkhornKnopp(hdl, this, d, e);
  cudaDeviceSynchronize();

  // Transform A = sign(A) .* sqrt(A) if 2-norm equilibration was performed,
  // or A = sign(A) .* A if the 1-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    __UnSetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        SqrtF<T>());
  } else {
    __UnSetSign<<<grid_size, cml::kBlockSize>>>(_data, sign, num_chars,
        IdentityF<T>());
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // Deal with last few entries if num_el is not a multiple of 8.
  if (num_el > num_chars * 8) {
    if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
      __UnSetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars,
          num_el - num_chars * 8, SqrtF<T>());
    } else {
      __UnSetSignSingle<<<1, 1>>>(_data + num_chars * 8, sign + num_chars,
          num_el - num_chars * 8, IdentityF<T>());
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  // Compute D := sqrt(D), E := sqrt(E), if 2-norm was equilibrated.
  if (kNormEquilibrate == kNorm2 || kNormEquilibrate == kNormFro) {
    thrust::transform(thrust::device_pointer_cast(d),
        thrust::device_pointer_cast(d + block.Rows()),
        thrust::device_pointer_cast(d), SqrtF<T>());
    thrust::transform(thrust::device_pointer_cast(e),
        thrust::device_pointer_cast(e + block.Cols()),
        thrust::device_pointer_cast(e), SqrtF<T>());
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();
  }

  // Compute A := D * A * E.
  MultDiag(d, e, block.Rows(), block.Cols(), _ord, _data);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERR();

  // Scale A to have norm of 1 (in the kNormNormalize norm).
  T normA = NormEst(hdl, kNormNormalize, *this);
  CUDA_CHECK_ERR();
  cudaDeviceSynchronize();
  cml::vector<T> a_vec = cml::vector_view_array(_data, num_el);
  cml::vector_scale(&a_vec, 1 / normA);
  cudaDeviceSynchronize();

  // Scale d and e to account for normalization of A.
  cml::vector<T> d_vec = cml::vector_view_array<T>(d, block.Rows());
  cml::vector<T> e_vec = cml::vector_view_array<T>(e, block.Cols());
  cml::vector_scale(&d_vec, 1 / sqrt(normA));
  cml::vector_scale(&e_vec, 1 / sqrt(normA));
  cudaDeviceSynchronize();

#ifdef DEBUG
  int kRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);

  T normD = mpih::dist_blas_nrm2(hdl, &d_vec);
  T normE = mpih::dist_blas_nrm2(hdl, &e_vec);
  MASTER(kRank) {
    BMARK_PRINT_T("equil_time", timer<double>() - t0);
    DEBUG_PRINTF("norm A = %e, normd = %e, norme = %e\n", normA, normD, normE);
  }
#endif

  cudaFree(sign);
  CUDA_CHECK_ERR();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Equilibration Helpers //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// Estimates norm of A. norm_type should either be kNorm2 or kNormFro.
template <typename T>
T NormEst(cublasHandle_t hdl, NormTypes norm_type,
          const MatrixDistDense<T>& A) {
  switch (norm_type) {
    case kNorm2: {
      return Norm2Est(hdl, &A);
    }
    case kNormFro: {
      MPI_Datatype t_type = (is_same<T,double>::value ?
                             MPI_DOUBLE :
                             MPI_FLOAT);
      const BlockMeta &block = A.Meta().block;

      const cml::vector<T> a = cml::vector_view_array(A.Data(),
          block.Rows() * block.Cols());
      T norm2;
      cml::blas_dot(hdl, &a, &a, &norm2);
      MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, t_type, MPI_SUM, MPI_COMM_WORLD);
      norm2 = sqrtf(norm2);
      return norm2 / std::sqrt(std::min(A.Rows(), A.Cols()));
    }
    case kNorm1:
      // 1-norm normalization doens't make make sense since it treats rows and
      // columns differently.
    default:
      ASSERT(false);
      return static_cast<T>(0.);
  }
}

// Performs A := D * A * E for A in row major
template <typename T>
void __global__ __MultRow(size_t m, size_t n, const T *d, const T *e, T *data) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t t = tid; t < m * n; t += gridDim.x * blockDim.x)
    data[t] *= d[t / n] * e[t % n];
}

// Performs A := D * A * E for A in col major
template <typename T>
void __global__ __MultCol(size_t m, size_t n, const T *d, const T *e, T *data) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t t = tid; t < m * n; t += gridDim.x * blockDim.x)
    data[t] *= d[t % m] * e[t / m];
}

template <typename T>
void MultDiag(const T *d, const T *e, size_t m, size_t n,
              typename MatrixDistDense<T>::Ord ord, T *data) {
  if (ord == MatrixDistDense<T>::ROW) {
    size_t grid_dim_row = cml::calc_grid_dim(m * n, cml::kBlockSize);
    __MultRow<<<grid_dim_row, cml::kBlockSize>>>(m, n, d, e, data);
  } else {
    size_t grid_dim_row = cml::calc_grid_dim(m * n, cml::kBlockSize);
    __MultCol<<<grid_dim_row, cml::kBlockSize>>>(m, n, d, e, data);
  }
}

/*
  Assume the entire block is on a single node and we need to transfer each
  block. In the future, we should support distributed matricies that start out
  across nodes.
 */
template <typename T>
void DistributeBlocks(const Schedule &s,
                      typename MatrixDistDense<T>::Ord ord,
                      const size_t m, const size_t n,
                      const T *orig_data, T *gpu_data) {
  int kRank, kNodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
  MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

  size_t width = (ord == MatrixDistDense<T>::ROW ? n : m);
  size_t height = (ord == MatrixDistDense<T>::ROW ? m : n);

  MPI_Request *request;
  MASTER(kRank) {
    int block_row = 0;
    int block_column = 0;
    int node = 0;

    request = new MPI_Request[kNodes];

    // We can optimize this by not random accessing the array and instead
    // figuring out which rows correspond to which node and then walking
    // through the rows thus performing sequential reads.
    if (ord == MatrixDistDense<T>::ROW) {
      for (int node = 0; node < kNodes; ++node) {
        const BlockMeta &block = s.At(node).block;
        for (size_t row = block.row_begin; row < block.row_end; ++row) {
          size_t offset = row * n + block.column_begin;
          size_t size = block.Cols();
          if (kRank == node) {
            size_t gpu_offset = (row - block.row_begin) * block.Cols();
            cudaMemcpy(gpu_data + gpu_offset, orig_data + offset,
                       size * sizeof(T), cudaMemcpyDefault);
          } else {
            MPI_Isend(orig_data + offset, size * sizeof(T), MPI_BYTE, node, 0,
                      MPI_COMM_WORLD, &request[node - 1]);
            if (row != block.row_end - 1)
              MPI_Request_free(&request[node - 1]);
          }
        }
      }
    } else {
      for (int node = 0; node < kNodes; ++node) {
        const BlockMeta &block = s.At(node).block;
        for (size_t col = block.column_begin; col < block.column_end; ++col) {
          size_t offset = col * m + block.row_begin;
          size_t size = block.row_end - block.row_begin;
          if (kRank == node) {
            size_t gpu_offset = (col - block.column_begin) * block.Rows();
            cudaMemcpy(gpu_data + offset, orig_data + offset, size * sizeof(T),
                       cudaMemcpyDefault);
          } else {
            MPI_Isend(orig_data + offset, size * sizeof(T), MPI_BYTE, node, 0,
                      MPI_COMM_WORLD, &request[node - 1]);
            if (col != block.column_end - 1)
              MPI_Request_free(&request[node - 1]);
          }
        }
      }
    }
  } else {
    const BlockMeta &block = s.At(kRank).block;

    size_t rows = block.row_end - block.row_begin;
    size_t columns = block.column_end - block.column_begin;
    if (ord == MatrixDistDense<T>::ROW) {
      for (size_t row = 0; row < rows; ++row) {
        size_t offset = row * columns;
        size_t size = columns;
        // MPI
        MPI_Recv(gpu_data + offset, size * sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    } else {
      for (size_t col = 0; col < columns; ++col) {
        size_t offset = col * rows;
        size_t size = rows;
        // MPI
        MPI_Recv(gpu_data + offset, size * sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

  MASTER(kRank) {
    MPI_Waitall(kNodes - 1, request, MPI_STATUSES_IGNORE);
    delete [] request;
  }
}

}  // namespace

  // Explicit template instantiation.
  template class MatrixDistDense<double>;
  template class MatrixDistDense<float>;

}  // namespace pogs
