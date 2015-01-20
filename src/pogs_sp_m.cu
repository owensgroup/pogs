#include <mpi.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "sinkhorn_knopp.cuh"

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#include "_interface_defs.h"
#include "cml/cml_linalg.cuh"
#include "cml/cml_spblas.cuh"
#include "cml/cml_spmat.cuh"
#include "cml/cml_vector.cuh"
#include "pogs.h"
//#include "timer.hpp"


// Apply operator to h.a and h.d.
template <typename T, typename Op>
struct ApplyOp: thrust::binary_function<FunctionObj<T>, FunctionObj<T>, T> {
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) { }
  __device__ FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x); h.d = binary_op(h.d, x);
    return h;
  }
};

template <typename T>
void SendFunctionObj(FunctionObj<T> &fo, int node, MPI_Request *request) {
  MPI_Isend(&fo.h, 1, MPI_INT, node, 0, MPI_COMM_WORLD, request);
  MPI_Isend(&fo.a, sizeof(T), MPI_BYTE, node, 0, MPI_COMM_WORLD, request);
  MPI_Isend(&fo.b, sizeof(T), MPI_BYTE, node, 0, MPI_COMM_WORLD, request);
  MPI_Isend(&fo.c, sizeof(T), MPI_BYTE, node, 0, MPI_COMM_WORLD, request);
  MPI_Isend(&fo.d, sizeof(T), MPI_BYTE, node, 0, MPI_COMM_WORLD, request);
  MPI_Isend(&fo.e, sizeof(T), MPI_BYTE, node, 0, MPI_COMM_WORLD, request);
}

template <typename T>
void RecvFunctionObj(std::vector<FunctionObj<T> > &fos) {
  Function h;
  T a, b, c, d, e;
  MPI_Recv(&h, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&a, sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&b, sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&c, sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&d, sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&e, sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  fos.push_back(FunctionObj<T>(h, a, b, c, d, e));
}

template<typename T>
inline int Allreduce(T *send, T *recv, int count, MPI_Op op, MPI_Comm comm);

template<>
inline int Allreduce(float *send,
                     float *recv,
                     int count,
                     MPI_Op op,
                     MPI_Comm comm) {
  return MPI_Allreduce(send, recv, count, MPI_FLOAT, op, comm);
}

template<>
inline int Allreduce(double *send,
                     double *recv,
                     int count,
                     MPI_Op op,
                     MPI_Comm comm) {
  return MPI_Allreduce(send, recv, count, MPI_DOUBLE, op, comm);
}

template<typename T>
inline int Allgather(T *send,
                     int send_count,
                     T *recv,
                     int recv_count,
                     MPI_Comm comm);

template<>
inline int Allgather(float *send,
                     int send_count,
                     float *recv,
                     int recv_count,
                     MPI_Comm comm) {
  return MPI_Allgather(send, send_count, MPI_FLOAT, recv, recv_count, MPI_FLOAT,
                       comm);
};

template<>
inline int Allgather(double *send,
                     int send_count,
                     double *recv,
                     int recv_count,
                     MPI_Comm comm) {
  return MPI_Allgather(send, send_count, MPI_DOUBLE, recv, recv_count,
                       MPI_DOUBLE, comm);
};

template <typename T, typename I, POGS_ORD O>
struct SendSubMatricesHelper {
  static void SendSubMatrices(PogsData<T, Sparse<T, I, O> > *pogs_data,
                              Sparse<T, I, O> &A) {
  }
};

template <typename T, typename I>
struct SendSubMatricesHelper<T, I, ROW> {
  static void SendSubMatrices(PogsData<T, Sparse<T, I, ROW> > *pogs_data,
                              Sparse<T, I, ROW> &A) {
    int kRank, kNodes;
    MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
    MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

    int m = pogs_data->m, n = pogs_data->n, nnz = pogs_data->A.nnz;
    int m_nodes = pogs_data->m_nodes, n_nodes = pogs_data->n_nodes;
    int m_sub = m / m_nodes, n_sub = n / n_nodes;

    int i_A = kRank / n_nodes; // Row, m
    int j_A = kRank % n_nodes; // Column, n

    if (kRank == 0) {
      MPI_Request *request = new MPI_Request[kNodes];
      MPI_Status row_status;
      int curr_i_A = 0;
      int curr_j_A = 0;
      int row = 0;
      int node = 0;
      I stripe_begin = 0;
      I i = 0;
      I row_end;

      // Distribute A_ij matrices and proximal operators to nodes
      for (curr_j_A = 0; curr_j_A < m_nodes; ++curr_j_A) {
        while (row < (curr_j_A + 1) * m_sub) {
          row_end = pogs_data->A.ptr[row + 1];

          for (curr_i_A = 0; curr_i_A < n_nodes; ++curr_i_A) {
            node = curr_i_A + curr_j_A * n_nodes;
            // Find this block's portion of the row
            while (i != row_end) {
              if (pogs_data->A.ind[i] > (curr_i_A + 1) * m_sub) {
                break;
              }
              i++;
            }
            MPI_Isend(pogs_data->A.val + stripe_begin,
                      (i - stripe_begin) * sizeof(T),
                      MPI_BYTE,
                      node,
                      0,
                      MPI_COMM_WORLD,
                      &request[node]);
            MPI_Isend(pogs_data->A.ind + stripe_begin,
                      (i - stripe_begin) * sizeof(I),
                      MPI_BYTE,
                      node,
                      0,
                      MPI_COMM_WORLD,
                      &request[node]);
            stripe_begin = i;
          }
          row++;
        }
      }

      node = 0;
      int f_increment = m_sub;
      int f_used = 0;
      // Send f operators
      for (curr_j_A = 0; curr_j_A < m_nodes - 1; ++curr_j_A) {
        // Send the rest of f to the last node
        for (; f_used < m_sub * (curr_j_A + 1); ++f_used) {
          SendFunctionObj(pogs_data->f[f_used], node, &request[node]);
        }
        node++;
      }

      for (; f_used < m; ++f_used) {
        SendFunctionObj(pogs_data->f[f_used], node, &request[node]);
      }

      node = 0;
      // Send proximal operators
      node = 0;
      int g_size = pogs_data->g.size();
      for (curr_j_A = 0; curr_j_A < m_nodes; ++curr_j_A) {
        for (int i = 0; i < g_size; ++i) {
          SendFunctionObj(pogs_data->g[i], node, &request[node]);
        }

        node++;
      }

      // Wait on all nodes except master node
      MPI_Waitall(kNodes - 1, request + 1, MPI_STATUSES_IGNORE);
      delete[] request;
    }

    T *val = A.val;
    I *ptr = A.ptr;
    I *ind = A.ind;
    int stripes = 0;
    int count;
    MPI_Status row_status;

    // Receive A_ij matrix 
    for (int stripes = 0; stripes < m_sub; ++stripes) {
      MPI_Recv(val, n_sub * sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &row_status);
      MPI_Get_count(&row_status, MPI_BYTE, &count);
      count /= sizeof(T);
      val += count;

      MPI_Recv(ind, n_sub * sizeof(I), MPI_BYTE, 0, MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &row_status);
      MPI_Get_count(&row_status, MPI_BYTE, &count);
      count /= sizeof(I);
      ind += count;

      // Set row/col indicies to be in local A_ij coordinates
      for (int i = 0; i < count; ++i) {
        ind[i] -= n_sub * j_A;
      }

      ptr[stripes + 1] = ptr[stripes] + count;
    }
    A.nnz = ptr[m_sub];

    // Receive f proximal operators
    int num_f;
    if (j_A == m_nodes - 1) {
      num_f = m - m_sub * (m_nodes - 1);
    } else {
      num_f = m_sub;
    }
    pogs_data->f.clear();
    pogs_data->f.reserve(num_f);
    for (int i = 0; i < num_f; ++i) {
      RecvFunctionObj(pogs_data->f);
    }

    // Receive g operators
    int num_g = pogs_data->n;
    pogs_data->g.clear();
    pogs_data->g.reserve(num_g);
    for (int i = 0; i < num_g; ++i) {
      RecvFunctionObj(pogs_data->g);
    }
  }
};

template <typename T, typename I>
struct SendSubMatricesHelper<T, I, COL> {
  static void SendSubMatrices(PogsData<T, Sparse<T, I, COL> > *pogs_data,
                              Sparse<T, I, COL> &A) {
    int kRank, kNodes;
    MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
    MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

    int m = pogs_data->m, n = pogs_data->n, nnz = pogs_data->A.nnz;
    int m_nodes = pogs_data->m_nodes, n_nodes = pogs_data->n_nodes;
    int m_sub = m / m_nodes, n_sub = n / n_nodes;

    int i_A = kRank % m_nodes;
    int j_A = kRank / m_nodes;

    if (kRank == 0) {
      MPI_Request *request = new MPI_Request[kNodes];
      MPI_Status col_status;
      int curr_i_A = 0;
      int curr_j_A = 0;
      int col = 0;
      int node = 0;
      I stripe_begin = 0;
      I i = 0;
      I col_end;

      // Distribute A_ij matrices and proximal operators to nodes
      for (curr_i_A = 0; curr_i_A < n_nodes; ++curr_i_A) {
        while (col < (curr_i_A + 1) * n_sub) {
          col_end = pogs_data->A.ptr[col + 1];

          for (curr_j_A = 0; curr_j_A < m_nodes; ++curr_j_A) {
            node = curr_j_A + curr_i_A * m_nodes;
            // Find this block's portion of the row
            while (i != col_end) {
              if (pogs_data->A.ind[i] > (curr_j_A + 1) * n_sub) {
                break;
              }
              i++;
            }
            MPI_Isend(pogs_data->A.val + stripe_begin,
                      (i - stripe_begin) * sizeof(T),
                      MPI_BYTE,
                      node,
                      0,
                      MPI_COMM_WORLD,
                      &request[node]);
            MPI_Isend(pogs_data->A.ind + stripe_begin,
                      (i - stripe_begin) * sizeof(I),
                      MPI_BYTE,
                      node,
                      0,
                      MPI_COMM_WORLD,
                      &request[node]);
            stripe_begin = i;
          }
          col++;
        }
      }

      node = 0;
      int g_size = pogs_data->g.size();
      for (curr_j_A = 0; curr_j_A < m_nodes; ++curr_j_A) {
        for (int i = 0; i < g_size; ++i) {
          SendFunctionObj(pogs_data->g[i], node, &request[node]);
        }

        node++;
      }

      node = 0;
      int f_increment = m_sub;
      int f_used = 0;
      // Send f operators
      for (curr_j_A = 0; curr_j_A < m_nodes - 1; ++curr_j_A) {
        // Send the rest of f to the last node
        for (; f_used < m_sub * (curr_j_A + 1); ++f_used) {
          SendFunctionObj(pogs_data->f[f_used], node, &request[node]);
        }
        node++;
      }

      for (; f_used < m; ++f_used) {
        SendFunctionObj(pogs_data->f[f_used], node, &request[node]);
      }
    
      // Wait on all nodes except master node
      MPI_Waitall(kNodes - 1, request + 1, MPI_STATUSES_IGNORE);
      delete[] request;
    }

    T *val = A.val;
    I *ptr = A.ptr;
    I *ind = A.ind;
    int stripes = 0;
    int count;
    MPI_Status col_status;

    // Receive A_ij matrix 
    for (int stripes = 0; stripes < n_sub; ++stripes) {
      MPI_Recv(val, m_sub * sizeof(T), MPI_BYTE, 0, MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &col_status);
      MPI_Get_count(&col_status, MPI_BYTE, &count);
      count /= sizeof(T);
      val += count;

      MPI_Recv(ind, m_sub * sizeof(I), MPI_BYTE, 0, MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &col_status);
      MPI_Get_count(&col_status, MPI_BYTE, &count);
      count /= sizeof(I);
      ind += count;

      // Set row/col indicies to be in local A_ij coordinates
      for (int i = 0; i < count; ++i) {
        ind[i] -= m_sub * i_A;
      }

      ptr[stripes] = ptr[stripes - 1] + count;
    }
    A.nnz = ptr[n_sub];

    // Receive g operators
    int num_g = pogs_data->n;
    pogs_data->g.clear();
    pogs_data->g.reserve(num_g);
    for (int i = 0; i < num_g; ++i) {
      RecvFunctionObj(pogs_data->g);
    }

    // Receive f proximal operators
    int num_f;
    if (j_A == m_nodes - 1) {
      num_f = m - m_sub * (m_nodes - 1);
    } else {
      num_f = m_sub;
    }
    pogs_data->f.clear();
    pogs_data->f.reserve(num_f);
    for (int i = 0; i < num_f; ++i) {
      RecvFunctionObj(pogs_data->f);
    }
  }
};

template<typename T, typename M>
void SendSubMatrices(PogsData<T, M> *pogs_data, M &A) {
  SendSubMatricesHelper<T, typename M::I_t, M::Ord>::SendSubMatrices(pogs_data, A);
}

// Proximal Operator Graph Solver.
template<typename T, typename M>
int Pogs(PogsData<T, M> *pogs_data) {
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin = static_cast<T>(1.05);
  const T kGamma = static_cast<T>(1.01);
  const T kTau = static_cast<T>(0.8);
  const T kAlpha = static_cast<T>(1.7);
  const T kKappa = static_cast<T>(0.9);
  const T kOne = static_cast<T>(1);
  const T kZero = static_cast<T>(0);
  const T kTol = static_cast<T>(1e-3);
  const T kRhoMax = static_cast<T>(1e4);
  const T kRhoMin = static_cast<T>(1e-4);
  const CBLAS_ORDER kOrd = M::Ord == ROW ? CblasRowMajor : CblasColMajor;

  // Setup MPI 
  int kLocalRank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  int kRank, kNodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
  MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

  // Set GPU device
  cudaSetDevice(kLocalRank);

  // Transfer pogs meta data
  //MPI_Bcast(&pogs_data->A.nnz, sizeof(M::I_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->m, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->m_nodes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->n_nodes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->rho, sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->abs_tol, sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->rel_tol, sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->max_iter, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->quiet, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->adaptive_rho, sizeof(bool), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->gap_stop, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->init_x, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&pogs_data->init_y, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

  // Extract meta data from pogs_data
  int m = pogs_data->m, n = pogs_data->n, nnz = pogs_data->A.nnz;
  int m_nodes = pogs_data->m_nodes, n_nodes = pogs_data->n_nodes;
  T rho = pogs_data->rho;

  int i_A, j_A;
  if (M::Ord == ROW) {
    i_A = kRank / n_nodes; // Row, m
    j_A = kRank % n_nodes; // Column, n
  } else {
    i_A = kRank % m_nodes; // Row, m
    j_A = kRank / m_nodes; // Column, n
  }
  Printf("Sub matrix position: i, j = %d, %d\n", i_A, j_A);

  // Setup sub matrix size
  int m_sub = m / m_nodes;
  int n_sub = n / n_nodes;
  Printf("Sub matrix size: %d x %d\n", m_sub, n_sub);

  M A_ij(new T[m_sub * n_sub],
         new typename M::I_t[M::Ord == ROW ? m_sub + 1 : n_sub + 1],
         new typename M::I_t[m_sub * n_sub],
         m_sub * n_sub);
  SendSubMatrices(pogs_data, A_ij);

  int nnz_sub = A_ij.nnz;
  Printf("Sub matrix nnz: %d\n", nnz_sub);
  
  thrust::device_vector<FunctionObj<T> > f = pogs_data->f;
  thrust::device_vector<FunctionObj<T> > g = pogs_data->g;

  int err = 0;

  // Average comm
  MPI_Comm avg_comm;
  MPI_Comm_split(MPI_COMM_WORLD, j_A, 0, &avg_comm);

  // Exchange comm
  MPI_Comm exch_comm;
  MPI_Comm_split(MPI_COMM_WORLD, i_A, 0, &exch_comm);
 
  // Create cuBLAS hdl.
  cublasHandle_t d_hdl;
  cublasCreate(&d_hdl);
  cusparseHandle_t s_hdl;
  cusparseCreate(&s_hdl);
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);
  cudaStream_t prox_s, aij_s;
  cudaStreamCreate(&prox_s);
  cudaStreamCreate(&aij_s);

  // Allocate data for ADMM variables.
  bool pre_process = true;
  cml::vector<T> de, z, zt;
  cml::vector<T> zprev = cml::vector_calloc<T>(m_sub + n_sub);
  cml::vector<T> z12 = cml::vector_calloc<T>(m_sub + n_sub);
  cml::vector<T> xh = cml::vector_calloc<T>(n_sub);
  cml::vector<T> xhtmp = cml::vector_calloc<T>(n_sub);
  cml::vector<T> x12tmp = cml::vector_calloc<T>(n_sub);
  cml::spmat<T, typename M::I_t, kOrd> A;
  if (pogs_data->factors.val != 0) {
    cudaMemcpy(&rho, pogs_data->factors.val, sizeof(T), cudaMemcpyDeviceToHost);
    pre_process = (rho == 0);
    if (pre_process)
      rho = pogs_data->rho;
    de = cml::vector_view_array(pogs_data->factors.val + 1, m + n);
    z = cml::vector_view_array(pogs_data->factors.val + 1 + m + n, m + n);
    zt = cml::vector_view_array(pogs_data->factors.val + 1 + 2 * (m + n),
        m + n);
    A = cml::spmat<T, typename M::I_t, kOrd>(
        pogs_data->factors.val + 1 + 3 * (m + n),
        pogs_data->factors.ind, pogs_data->factors.ptr, m, n,
        pogs_data->factors.nnz);
  } else {
    de = cml::vector_calloc<T>(m_sub + n_sub);
    z = cml::vector_calloc<T>(m_sub + n_sub);
    zt = cml::vector_calloc<T>(m_sub + n_sub);
    A = cml::spmat_alloc<T, typename M::I_t, kOrd>(m_sub, n_sub, nnz_sub);
  }

  if (de.data == 0 || z.data == 0 || zt.data == 0 || zprev.data == 0 ||
      z12.data == 0 || A.val == 0 || A.ind == 0 || A.ptr == 0) {
    err = 1;
  }

  // Create views for x and y components.
  cml::vector<T> d = cml::vector_subvector(&de, 0, m_sub);
  cml::vector<T> e = cml::vector_subvector(&de, m_sub, n_sub);
  cml::vector<T> x = cml::vector_subvector(&z, 0, n_sub);
  cml::vector<T> y = cml::vector_subvector(&z, n_sub, m_sub);
  cml::vector<T> x12 = cml::vector_subvector(&z12, 0, n_sub);
  cml::vector<T> y12 = cml::vector_subvector(&z12, n_sub, m_sub);
  cml::vector<T> xt = cml::vector_subvector(&zt, 0, n_sub);
  cml::vector<T> yt = cml::vector_subvector(&zt, n_sub, m_sub);
  cml::vector<T> xprev = cml::vector_subvector(&zprev, 0, n_sub);
  cml::vector<T> yprev = cml::vector_subvector(&zprev, n_sub, m_sub);

  // For transfering values using mpi when we don't have OMPI CUDA
#ifndef __OMPI_CUDA__
  std::vector<T> xh_h(n_sub);
  std::vector<T> xhtmp_h(n_sub);
#else
  cml::matrix<T, CblasRowMajor> gather_buf =
    cml::matrix_calloc<T, CblasRowMajor>(kNodes, n_sub);
  cml::vector<T> identity = cml::vector_alloc<T>(kNodes);
  cml::vector_set_all(&identity, kOne);
#endif

  if (pre_process && !err) {
    cml::spmat_memcpy(s_hdl, &A, A_ij.val, A_ij.ind, A_ij.ptr);
    // Avoid equilibration for now
    //err = sinkhorn_knopp::Equilibrate(s_hdl, d_hdl, descr, &A, &d, &e);

    if (!err) {
      // TODO: Issue warning if x == NULL or y == NULL
      // Initialize x and y from x0 or/and y0
      if (pogs_data->init_x && !pogs_data->init_y && pogs_data->x) {
        cml::vector_memcpy(&x, pogs_data->x);
        cml::vector_div(&x, &e);
        cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, kOne,
            &A, &x, kZero, &y);
      } else if (pogs_data->init_y && !pogs_data->init_x && pogs_data->y) {
        cml::vector_memcpy(&y, pogs_data->y);
        cml::vector_mul(&y, &d);
        cml::vector_set_all(&x, kZero);
        cml::spblas_solve(s_hdl, d_hdl, descr, &A, static_cast<T>(1e-4), &y, &x,
            static_cast<T>(1e-6), 100, true);
        cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, kOne,
            &A, &x, kZero, &y);
      } else if (pogs_data->init_x && pogs_data->init_y &&
          pogs_data->x && pogs_data->y) {
        cml::vector_memcpy(&y, pogs_data->y);
        cml::vector_mul(&y, &d);
        cml::vector_memcpy(&x, pogs_data->x);
        cml::vector_div(&x, &e);
        cml::vector_memcpy(&x12, &x);
        cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, -kOne,
            &A, &x, kOne, &y);
        cml::spblas_solve(s_hdl, d_hdl, descr, &A, kOne, &y, &x12,
            static_cast<T>(1e-6), 100, true);
        cml::blas_axpy(d_hdl, -kOne, &x12, &x);
        cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, kOne,
            &A, &x, kZero, &y);
      }
    }
  }

  // todo(abp): figure out how e and d work with Block splitting

  // Scale f and g to account for diagonal scaling e and d.
  /*
  if (!err) {
    thrust::transform(f.begin(), f.end(), thrust::device_pointer_cast(d.data),
        f.begin(), ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
    thrust::transform(g.begin(), g.end(), thrust::device_pointer_cast(e.data),
        g.begin(), ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));
  }
  */

  // Signal start of execution.
  if (!pogs_data->quiet)
    Printf("   #      res_pri    eps_pri   res_dual   eps_dual"
           "        gap    eps_gap  objective\n");

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n_sub)) * pogs_data->abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m_sub)) * pogs_data->abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int kd = 0, ku = 0;
  bool converged = false;

  //double t = timer<double>();
  for (unsigned int k = 0; !err; ++k) {
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    cml::blas_axpy(d_hdl, -kOne, &zt, &z);
    ProxEval(g, rho, xh.data, xh.stride, x12.data, x12.stride);
    ProxEval(f, rho, y.data, y.stride, y12.data, y12.stride);

    
    // Compute dual variable.
    T nrm_r = 0, nrm_s = 0, gap, z_nrm, z12_nrm, temp;
    cml::blas_axpy(d_hdl, -kOne, &z12, &z);
    cml::blas_dot(d_hdl, &z, &z12, &gap);
    gap = std::abs(gap);
    pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);

    // Calculate global z norm
    z_nrm = cml::blas_dot(d_hdl, &y, &y);
    Allreduce(&z_nrm, &temp, 1, MPI_SUM, MPI_COMM_WORLD);
    z_nrm = sqrtf(cml::blas_dot(d_hdl, &xh, &xh) + temp);

    // Calculate global z12 norm
    z12_nrm = cml::blas_dot(d_hdl, &y12, &y12);
    Allreduce(&z12_nrm, &temp, 1, MPI_SUM, MPI_COMM_WORLD);
    z12_nrm = sqrtf(cml::blas_dot(d_hdl, &x12, &x12) + temp);
    
    T eps_gap = std::sqrt(static_cast<T>(m + n)) * pogs_data->abs_tol +
        pogs_data->rel_tol * z_nrm * z12_nrm;
    T eps_pri = sqrtm_atol + pogs_data->rel_tol * z12_nrm;
    T eps_dua = sqrtn_atol + pogs_data->rel_tol * rho * z_nrm;

    if (converged || k == pogs_data->max_iter)
      break;

    // Project onto A_ij
    cml::vector_memcpy(&y, &y12);
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, -kOne, &A,
        &x12, kOne, &y);

    // Compute primal residual
    nrm_r = cml::blas_dot(d_hdl, &y, &y);
    Allreduce(&nrm_r, &nrm_s, 1, MPI_SUM, MPI_COMM_WORLD);
    nrm_r = sqrtf(nrm_s);
    nrm_s = 0;

    cml::vector_set_all(&x, kZero);
    cml::spblas_solve(s_hdl, d_hdl, descr, &A, kOne, &y, &x, kTol, 5, true);
    cml::blas_axpy(d_hdl, kOne, &x12, &x);
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, kOne, &A,
        &x, kZero, &y);

    // Apply over relaxation.
    cml::blas_scal(d_hdl, kAlpha, &z);
    cml::blas_axpy(d_hdl, kOne - kAlpha, &zprev, &z);

    // Update dual variable.
    cml::blas_axpy(d_hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(d_hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(d_hdl, -kOne, &z, &zt);

    // Average
    if (m_nodes > 1) {
      cml::vector_memcpy(&xhtmp, &x);
      cml::blas_axpy(d_hdl, -kOne, &xt, &xhtmp);
#ifndef __OMPI_CUDA__
      cudaMemcpy(xhtmp_h.data(), xhtmp.data, xhtmp.size,
                 cudaMemcpyDeviceToHost);
      Allreduce(xhtmp_h.data(), xh_h.data(), xh_h.size(), MPI_SUM,
                MPI_COMM_WORLD);
      cudaMemcpy(xh.data, xh_h.data(), xh_h.size(), cudaMemcpyHostToDevice);
#else
      Allgather(xhtmp.data, xhtmp.size, gather_buf.data, xhtmp.size,
        MPI_COMM_WORLD);
      cml::blas_gemv(d_hdl, CUBLAS_OP_T, kOne, &gather_buf, &identity, kZero,
        &xh);
#endif
      cml::blas_scal(d_hdl, 1.0 / m_nodes, &xh);
    }

    bool exact = false;
    cml::blas_axpy(d_hdl, -kOne, &zprev, &z12);
    cml::blas_axpy(d_hdl, -kOne, &z, &zprev);

    // Calculate global dual residual norm approximation
    z_nrm = cml::blas_dot(d_hdl, &yprev, &yprev);
    Allreduce(&z_nrm, &temp, 1, MPI_SUM, MPI_COMM_WORLD);
    z_nrm = sqrtf(cml::blas_dot(d_hdl, &xprev, &xprev) + temp);

    nrm_s = rho * z_nrm;
    if (nrm_r < eps_pri && nrm_s < eps_dua) {
      // Calculate global A'y12 + x12
      cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_TRANSPOSE, descr, kOne, &A,
          &y12, kOne, &x12);
      nrm_s = cml::blas_dot(d_hdl, &x12, &x12);
      Allreduce(&nrm_s, &temp, 1, MPI_SUM, MPI_COMM_WORLD);
      nrm_s = rho * sqrtf(temp);
      exact = true;
    }

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!pogs_data->gap_stop || gap < eps_gap);
    if (!pogs_data->quiet && (k % 10 == 0 || converged))
      Printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, pogs_data->optval);

    // Rescale rho.
    if (pogs_data->adaptive_rho && !converged) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        if (rho < kRhoMax) {
          rho *= delta;
          cml::blas_scal(d_hdl, 1 / delta, &zt);
          delta = kGamma * delta;
          ku = k;
          Printf("+ rho %e\n", rho);
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (rho > kRhoMin) {
          rho /= delta;
          cml::blas_scal(d_hdl, delta, &zt);
          delta = kGamma * delta;
          kd = k;
          Printf("- rho %e\n", rho);
        }
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = std::max(delta / kGamma, kDeltaMin);
      }
    }
  }
  //Printf("TIME = %e\n", timer<double>() - t);

  // Scale x, y and l for output.
  cml::vector_div(&y12, &d);
  cml::vector_mul(&x12, &e);
  cml::vector_mul(&y, &d);
  cml::blas_scal(d_hdl, rho, &y);

  // Copy results to output.
  if (pogs_data->y != 0 && !err)
    cml::vector_memcpy(pogs_data->y, &y12);
  if (pogs_data->x != 0 && !err)
    cml::vector_memcpy(pogs_data->x, &x12);
  if (pogs_data->l != 0 && !err)
    cml::vector_memcpy(pogs_data->l, &y);

  #ifdef __OMPI_CUDA__
  cml::matrix_free(&gather_buf);
  cml::vector_free(&identity);
  #endif

  // Store rho and free memory.
  if (pogs_data->factors.val != 0 && !err) {
    cudaMemcpy(pogs_data->factors.val, &rho, sizeof(T), cudaMemcpyHostToDevice);
    cml::vector_memcpy(&z, &zprev);
  } else {
    cml::vector_free(&de);
    cml::vector_free(&z);
    cml::vector_free(&zt);
    cml::spmat_free(&A);
  }
  cml::vector_free(&xh);
  cml::vector_free(&xhtmp);
  cml::vector_free(&z12);
  cml::vector_free(&zprev);
  cudaStreamDestroy(aij_s);
  cudaStreamDestroy(prox_s);
  cusparseDestroyMatDescr(descr);
  cusparseDestroy(s_hdl);
  cublasDestroy(d_hdl);
  delete A_ij.val;
  delete A_ij.ptr;
  delete A_ij.ind;

  return err;
}

template <typename T, typename I, POGS_ORD O>
int AllocSparseFactors(PogsData<T, Sparse<T, I, O> > *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n, nnz = pogs_data->A.nnz;
  size_t flen = 1 + 3 * (n + m) + nnz;
  printf("flen = %lu\n", flen);

  Sparse<T, I, O>& A = pogs_data->factors;
  A.val = 0;
  A.ind = 0;
  A.ptr = 0;
  A.nnz = nnz;

  cudaError_t err = cudaMalloc(&A.val, 2 * flen * sizeof(T));
  if (err == cudaSuccess)
    err = cudaMemset(A.val, 0, 2 * flen * sizeof(T));
  if (err != cudaSuccess) {
    cudaFree(A.val);
    return 1;
  }

  err = cudaMalloc(&A.ind, 2 * nnz * sizeof(I));
  if (err == cudaSuccess)
    err = cudaMemset(A.ind, 0, 2 * nnz * sizeof(I));
  if (err != cudaSuccess) {
    cudaFree(A.ind);
    cudaFree(A.val);
    return 1;
  }

  err = cudaMalloc(&A.ptr, (m + n + 2) * sizeof(I));
  if (err == cudaSuccess)
    err = cudaMemset(A.ptr, 0, (m + n + 2) * sizeof(I));
  if (err != cudaSuccess) {
    cudaFree(A.ptr);
    cudaFree(A.ind);
    cudaFree(A.val);
    return 1;
  }

  return 0;
}

template <typename T, typename I, POGS_ORD O>
void FreeSparseFactors(PogsData<T, Sparse<T, I,O> > *pogs_data) {
  Sparse<T, I, O> &A = pogs_data->factors;
  cudaFree(A.ptr);
  cudaFree(A.ind);
  cudaFree(A.val);

  A.val = 0;
  A.ind = 0;
  A.ptr = 0;
}


// Declarations.
template int Pogs<double, Sparse<double, int, COL> >
    (PogsData<double, Sparse<double, int, COL> > *);
template int Pogs<double, Sparse<double, int, ROW> >
    (PogsData<double, Sparse<double, int, ROW> > *);
template int Pogs<float, Sparse<float, int, COL> >
    (PogsData<float, Sparse<float, int, COL> > *);
template int Pogs<float, Sparse<float, int, ROW> >
    (PogsData<float, Sparse<float, int, ROW> > *);

template int AllocSparseFactors<double, int, ROW>
    (PogsData<double, Sparse<double, int, ROW> > *);
template int AllocSparseFactors<double, int, COL>
    (PogsData<double, Sparse<double, int, COL> > *);
template int AllocSparseFactors<float, int, ROW>
    (PogsData<float, Sparse<float, int, ROW> > *);
template int AllocSparseFactors<float, int, COL>
    (PogsData<float, Sparse<float, int, COL> > *);

template void FreeSparseFactors<double, int, ROW>
    (PogsData<double, Sparse<double, int, ROW> > *);
template void FreeSparseFactors<double, int, COL>
    (PogsData<double, Sparse<double, int, COL> > *);
template void FreeSparseFactors<float, int, ROW>
    (PogsData<float, Sparse<float, int, ROW> > *);
template void FreeSparseFactors<float, int, COL>
    (PogsData<float, Sparse<float, int, COL> > *);
