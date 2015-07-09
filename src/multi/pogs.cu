#include "pogs.h"

#include <mpi.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>

#include "cml/cml_blas.cuh"
#include "cml/cml_vector.cuh"
#include "interface_defs.h"
#include "matrix/matrix_dist.h"
#include "matrix/matrix_dist_dense.h"
//#include "matrix/matrix_dist_sparse.h"
#include "projector/projector.h"
#include "projector/projector_direct.h"
#include "projector/projector_cgls.h"
#include "util.h"
#include "mpi_helper.h"

#include "timer.h"

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace pogs {

namespace {


template <typename T, typename Op>
struct ApplyOp: thrust::binary_function<FunctionObj<T>, FunctionObj<T>, T> {
  Op binary_op;
  ApplyOp(Op binary_op) : binary_op(binary_op) { }
  __host__ __device__ FunctionObj<T> operator()(FunctionObj<T> &h, T x) {
    h.a = binary_op(h.a, x);
    h.d = binary_op(h.d, x);
    h.e = binary_op(binary_op(h.e, x), x);
    return h;
  }
};

}  // namespace

template <typename T, typename M, typename P>
Pogs<T, M, P>::Pogs(const M &A)
    : _A(A), _P(_A),
      _de(0), _z(0), _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _done_init(false),
      _x(0), _y(0), _mu(0), _lambda(0), _optval(static_cast<T>(0.)),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _gap_stop(kGapStop),
      _init_x(false), _init_lambda(false) {
  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
}

template <typename T, typename M, typename P>
int Pogs<T, M, P>::_Init() {
  DEBUG_EXPECT(!_done_init);
  if (_done_init)
    return 1;
  _done_init = true;

  const BlockMeta &block = _A.Meta().block;

  size_t m = _A.Rows();
  size_t n = _A.Cols();
  size_t m_block = block.Rows();
  size_t n_block = block.Cols();

  cudaSetDevice(_A.Meta().gpu_indicies[0]);

  cudaMalloc(&_de, (m_block + n_block) * sizeof(T));
  cudaMalloc(&_z, (m_block + n_block) * sizeof(T));
  cudaMalloc(&_zt, (m_block + n_block) * sizeof(T));
  cudaMemset(_de, 0, (m_block + n_block) * sizeof(T));
  cudaMemset(_z, 0, (m_block + n_block) * sizeof(T));
  cudaMemset(_zt, 0, (m_block + n_block) * sizeof(T));
  CUDA_CHECK_ERR();

  _A.Init();
  _A.Equil(_de, _de + m_block);
  _P.Init();
  CUDA_CHECK_ERR();

  return 0;
}

template <typename T>
void DistributeProximals(const Schedule &schedule,
                         const std::vector<FunctionObj<T> > &f,
                         const std::vector<FunctionObj<T> > &g,
                         std::vector<FunctionObj<T> > &local_f,
                         std::vector<FunctionObj<T> > &local_g) {
  int kRank, kNodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
  MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

  size_t fobj_size = sizeof(FunctionObj<T>);

  MPI_Request *request;
  MASTER(kRank) {
    request = new MPI_Request[kNodes];

    for (int node = 0; node < kNodes; ++node) {
      const BlockMeta &meta = schedule.At(node).block;
      // Send f operators for rows
      size_t row_size = meta.row_end - meta.row_begin;
      size_t row_offset = meta.row_begin;
      MPI_Isend(f.data() + row_offset, fobj_size * row_size, MPI_BYTE, node, 0,
                MPI_COMM_WORLD, &request[node]);

      // Send g operators for columns
      size_t col_size = meta.column_end - meta.column_begin;
      size_t col_offset = meta.column_begin;
      MPI_Isend(g.data() + col_offset, fobj_size * col_size, MPI_BYTE, node, 0,
                MPI_COMM_WORLD, &request[node]);
    }
  }

  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);

  const BlockMeta &meta = schedule.At(kRank).block;

  // Receive f operators for rows
  size_t row_size = meta.row_end - meta.row_begin;
  local_f.resize(row_size);
  MPI_Recv(local_f.data(), fobj_size * row_size, MPI_BYTE, 0, MPI_ANY_TAG,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Receive g operators for rows
  size_t col_size = meta.column_end - meta.column_begin;
  local_g.resize(col_size);
  MPI_Recv(local_g.data(), fobj_size * col_size, MPI_BYTE, 0, MPI_ANY_TAG,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MASTER(kRank) {
    MPI_Waitall(kNodes, request, MPI_STATUSES_IGNORE);
    delete [] request;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

template <typename T, typename M, typename P>
PogsStatus Pogs<T, M, P>::Solve(const std::vector<FunctionObj<T> > &f,
                                const std::vector<FunctionObj<T> > &g) {
  double t0 = timer<double>();
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin   = static_cast<T>(1.05);
  const T kGamma      = static_cast<T>(1.01);
  const T kTau        = static_cast<T>(0.8);
  const T kAlpha      = static_cast<T>(1.7);
  const T kRhoMin     = static_cast<T>(1e-4);
  const T kRhoMax     = static_cast<T>(1e4);
  const T kKappa      = static_cast<T>(0.4);
  const T kOne        = static_cast<T>(1.0);
  const T kZero       = static_cast<T>(0.0);
  const T kProjTolMax = static_cast<T>(1e-8);
  const T kProjTolMin = static_cast<T>(1e-2);
  const T kProjTolPow = static_cast<T>(1.3);
  const T kProjTolIni = static_cast<T>(1e-5);
  bool use_exact_stop = true;

  int kRank, kNodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &kRank);
  MPI_Comm_size(MPI_COMM_WORLD, &kNodes);

  const BlockMeta &block = _A.Meta().block;

  // Average comm
  MPI_Comm avg_comm, exch_comm;
  printf("%d before main splits\n", kRank);
  //MPI_Comm_split(MPI_COMM_WORLD, block.row, 0, &exch_comm);
  MPI_Comm_dup(MPI_COMM_SELF, &exch_comm);
  printf("%d after exch splits\n", kRank);
  //MPI_Comm_split(MPI_COMM_WORLD, block.column, 0, &avg_comm);
  MPI_Comm_dup(MPI_COMM_WORLD, &avg_comm);
  printf("%d after avg splits\n", kRank);

  // Initialize Projector P and Matrix A.
  if (!_done_init)
    _Init();


  MPI_Datatype t_type = mpih::MPIDTypeFromT<T>();

  double prox_time = timer<double>();
  std::vector<FunctionObj<T> > local_f, local_g;
  DistributeProximals(_A.GetSchedule(), f, g, local_f, local_g);
  MASTER(kRank) {
    BMARK_PRINT_T("dist_prox_time", timer<double>() - prox_time);
  }

  // Extract values from pogs_data
  size_t m = _A.Rows();
  size_t n = _A.Cols();
  size_t m_block = block.Rows();
  size_t n_block = block.Cols();
  thrust::device_vector<FunctionObj<T> > f_gpu = local_f;
  thrust::device_vector<FunctionObj<T> > g_gpu = local_g;

  // Create cuBLAS handle.
  cublasHandle_t hdl;
  cublasCreate(&hdl);
  CUDA_CHECK_ERR();

  // Allocate data for ADMM variables.
  cml::vector<T> de    = cml::vector_view_array(_de, m_block + n_block);
  cml::vector<T> z     = cml::vector_view_array(_z, m_block + n_block);
  cml::vector<T> zt    = cml::vector_view_array(_zt, m_block + n_block);
  cml::vector<T> zprev = cml::vector_calloc<T>(m_block + n_block);
  cml::vector<T> ztemp = cml::vector_calloc<T>(m_block + n_block);
  cml::vector<T> z12   = cml::vector_calloc<T>(m_block + n_block);
  cml::vector<T> x_avg = cml::vector_calloc<T>(n_block);
  cml::vector<T> x_avg_temp = cml::vector_calloc<T>(n_block);
  CUDA_CHECK_ERR();

  // Create views for x and y components.
  cml::vector<T> d     = cml::vector_subvector(&de, 0, m_block);
  cml::vector<T> e     = cml::vector_subvector(&de, m_block, n_block);
  cml::vector<T> x     = cml::vector_subvector(&z, 0, n_block);
  cml::vector<T> y     = cml::vector_subvector(&z, n_block, m_block);
  cml::vector<T> x12   = cml::vector_subvector(&z12, 0, n_block);
  cml::vector<T> y12   = cml::vector_subvector(&z12, n_block, m_block);
  cml::vector<T> xt    = cml::vector_subvector(&zt, 0, n_block);
  cml::vector<T> yt    = cml::vector_subvector(&zt, n_block, m_block);
  cml::vector<T> xprev = cml::vector_subvector(&zprev, 0, n_block);
  cml::vector<T> yprev = cml::vector_subvector(&zprev, n_block, m_block);
  cml::vector<T> xtemp = cml::vector_subvector(&ztemp, 0, n_block);
  cml::vector<T> ytemp = cml::vector_subvector(&ztemp, n_block, m_block);
  CUDA_CHECK_ERR();

  // Scale f and g to account for diagonal scaling e and d.
  thrust::transform(f_gpu.begin(), f_gpu.end(),
      thrust::device_pointer_cast(d.data), f_gpu.begin(),
      ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
  thrust::transform(g_gpu.begin(), g_gpu.end(),
      thrust::device_pointer_cast(e.data), g_gpu.begin(),
      ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));

  thrust::device_vector<T> over_m(n_block, _A.GetSchedule().MBlocks());
  thrust::transform(g_gpu.begin(), g_gpu.end(),
                    over_m.begin(), g_gpu.begin(),
                    ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
  CUDA_CHECK_ERR();

  // Initialize (x, lambda) from (x0, lambda0).
  if (_init_x) {
    cml::vector_memcpy(&xtemp, _x);
    cml::vector_div(&xtemp, &e);
    _A.Mul('n', kOne, xtemp.data, kZero, ytemp.data);
    cudaDeviceSynchronize();
    cml::vector_memcpy(&z, &ztemp);
    CUDA_CHECK_ERR();
  }
  if (_init_lambda) {
    cml::vector_memcpy(&ytemp, _lambda);
    cml::vector_div(&ytemp, &d);
    _A.Mul('t', -kOne, ytemp.data, kZero, xtemp.data);
    cudaDeviceSynchronize();
    cml::blas_scal(hdl, -kOne / _rho, &ztemp);
    cml::vector_memcpy(&zt, &ztemp);
    CUDA_CHECK_ERR();
  }

  // Make an initial guess for (x0 or lambda0).
  if (_init_x && !_init_lambda) {
    // Alternating projections to satisfy
    //   1. \lambda \in \partial f(y), \mu \in \partial g(x)
    //   2. \mu = -A^T\lambda
    cml::vector_set_all(&zprev, kZero);
    for (unsigned int i = 0; i < kInitIter; ++i) {
      ProjSubgradEval(g_gpu, xprev.data, x.data, xtemp.data);
      ProjSubgradEval(f_gpu, yprev.data, y.data, ytemp.data);
      _P.Project(xtemp.data, ytemp.data, kOne, xprev.data, yprev.data,
          kProjTolIni);
      cudaDeviceSynchronize();
      CUDA_CHECK_ERR();
      cml::blas_axpy(hdl, -kOne, &ztemp, &zprev);
      cml::blas_scal(hdl, -kOne, &zprev);
    }
    // xt = -1 / \rho * \mu, yt = -1 / \rho * \lambda.
    cml::vector_memcpy(&zt, &zprev);
    cml::blas_scal(hdl, -kOne / _rho, &zt);
  } else if (_init_lambda && !_init_x) {
    ASSERT(false);
  }
  _init_x = _init_lambda = false;

  // Save initialization time.
  double time_init = timer<double>() - t0;

  // Signal start of execution.
  MASTER(kRank) {
    if (_verbose > 0) {
      Printf(__HBAR__
             "           POGS v%s - Proximal Graph Solver                      \n"
             "           (c) Christopher Fougner, Stanford University 2014-2015\n",
             POGS_VERSION.c_str());
    }
    if (_verbose > 1) {
      Printf(__HBAR__
             " Iter | pri res | pri tol | dua res | dua tol |   gap   | eps gap |"
             " pri obj\n" __HBAR__);
    }
  }

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * _abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * _abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * _abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int k = 0u, kd = 0u, ku = 0u;
  bool converged = false;
  T nrm_r, nrm_s, gap, eps_gap, eps_pri, eps_dua;

  for (;; ++k) {
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    cml::blas_axpy(hdl, -kOne, &zt, &z);
    ProxEval(g_gpu, _rho, x_avg.data, x12.data);
    ProxEval(f_gpu, _rho, y.data, y12.data);
    CUDA_CHECK_ERR();


    // Compute gap, optval, and tolerances.
    cml::blas_axpy(hdl, -kOne, &z12, &z);

    /*cml::blas_dot(hdl, &z, &z12, &gap);
    gap = std::abs(gap);
    eps_gap = sqrtmn_atol + _rel_tol * cml::blas_nrm2(hdl, &z) *
    cml::blas_nrm2(hdl, &z12); */
    eps_pri = sqrtm_atol + _rel_tol * mpih::dist_blas_nrm2(hdl, &y12);
    eps_dua = _rho * (sqrtn_atol + _rel_tol * mpih::dist_blas_nrm2(hdl, &x));
    CUDA_CHECK_ERR();

    // Apply over relaxation.
    cml::vector_memcpy(&ztemp, &zt);
    cml::blas_axpy(hdl, kAlpha, &z12, &ztemp);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &ztemp);
    CUDA_CHECK_ERR();

    // Project onto y = Ax.
    T proj_tol = kProjTolMin / std::pow(static_cast<T>(k + 1), kProjTolPow);
    proj_tol = std::max(proj_tol, kProjTolMax);
    _P.Project(xtemp.data, ytemp.data, kOne, x.data, y.data, proj_tol);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERR();

    ///////////
    // TODO: Not sure how to handle this because zprev is different for each
    // block so we can't really compute a global norm easily
    //////////
    // Calculate residuals.
    // cml::vector_memcpy(&ztemp, &zprev);
    // cml::blas_axpy(hdl, -kOne, &z, &ztemp);
    // cudaDeviceSynchronize();
    // nrm_s = mpih::dist_blas_dot(hdl, &ytemp, &ytemp);
    // nrm_s = _rho * sqrtf(nrm_s + cml::blas_dot(hdl, &xtemp, &xtemp));

    // cml::vector_memcpy(&ztemp, &z12);
    // cml::blas_axpy(hdl, -kOne, &z, &ztemp);
    // cudaDeviceSynchronize();
    // nrm_r = mpih::dist_blas_dot(hdl, &ytemp, &ytemp);
    // nrm_r = sqrtf(nrm_r + cml::blas_dot(hdl, &xtemp, &xtemp));

    // Calculate exact residuals only if necessary.
    bool exact = false;
    if (true || (nrm_r < eps_pri && nrm_s < eps_dua) || use_exact_stop) {
      cml::vector_memcpy(&ztemp, &z12);
      _A.BlockMul('n', kOne, x12.data, -kOne, ytemp.data);
      cudaDeviceSynchronize();
      nrm_r = mpih::dist_blas_nrm2(hdl, &ytemp);
      if ((nrm_r < eps_pri) || use_exact_stop) {
        cml::vector_memcpy(&ztemp, &z12);
        cml::blas_axpy(hdl, kOne, &zt, &ztemp);
        cml::blas_axpy(hdl, -kOne, &zprev, &ztemp);
        _A.BlockMul('t', kOne, ytemp.data, kOne, xtemp.data);
        cudaDeviceSynchronize();
        nrm_s = _rho * mpih::dist_blas_nrm2(hdl, &xtemp);
        exact = true;
      }
    }
    CUDA_CHECK_ERR();

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!_gap_stop || gap < eps_gap);
    if (_verbose > 2 && k % 10  == 0 ||
        _verbose > 1 && k % 100 == 0 ||
        _verbose > 1 && converged) {
      T optval = FuncEval(f_gpu, y12.data);
      MPI_Allreduce(MPI_IN_PLACE, &optval, 1, t_type, MPI_SUM, MPI_COMM_WORLD);

      // Unscale
      thrust::transform(g_gpu.begin(), g_gpu.end(),
                        over_m.begin(), g_gpu.begin(),
                        ApplyOp<T, thrust::multiplies<T> >
                        (thrust::multiplies<T>()));
      optval = optval + FuncEval(g_gpu, x12.data);
      // Rescale
      thrust::transform(g_gpu.begin(), g_gpu.end(),
                        over_m.begin(), g_gpu.begin(),
                        ApplyOp<T, thrust::divides<T> >
                        (thrust::divides<T>()));
      if (kRank == 0) {
        Printf("%5d : %.2e  %.2e  %.2e  %.2e  %.2e  %.2e % .2e\n",
               k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, optval);
      }
    }

    // Break if converged or there are nans
    if (converged || k == _max_iter - 1)// || cml::vector_any_isnan(&zt))
      break;

    // Update dual variable.
    cml::blas_axpy(hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(hdl, -kOne, &z, &zt);
    CUDA_CHECK_ERR();

    // Rescale rho.
    if (_adaptive_rho) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        if (_rho < kRhoMax) {
          _rho *= delta;
          cml::blas_scal(hdl, 1 / delta, &zt);
          delta = kGamma * delta;
          ku = k;
          if (_verbose > 3)
            Printf("+ rho %e\n", _rho);
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (_rho > kRhoMin) {
          _rho /= delta;
          cml::blas_scal(hdl, delta, &zt);
          delta = kGamma * delta;
          kd = k;
          if (_verbose > 3)
            Printf("- rho %e\n", _rho);
        }
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = kDeltaMin;
      }
      CUDA_CHECK_ERR();
    }

    // Perform average operation over x values
    cml::vector_memcpy(&x_avg, &x);
    cml::blas_axpy(hdl, -kOne, &xt, &x_avg);
    MASTER(kRank) {
      mpih::Reduce(hdl, x_avg.data, x_avg_temp.data, x_avg.size, MPI_SUM, 0,
                   MPI_COMM_WORLD);
      cml::vector_memcpy(&x_avg, &x_avg_temp);
      cml::blas_scal(hdl, 1.0 / _A.GetSchedule().MBlocks(), &x_avg);
    } else {
      mpih::Reduce(hdl, x_avg.data, x_avg.data, x_avg.size, MPI_SUM, 0,
                   MPI_COMM_WORLD);
    }
    MPI_Bcast(x_avg.data, x_avg.size, t_type, 0, MPI_COMM_WORLD);
  }

  // Reverse division by number of M blocks so we can compute global opt value
  thrust::transform(g_gpu.begin(), g_gpu.end(),
                    over_m.begin(), g_gpu.begin(),
                    ApplyOp<T, thrust::multiplies<T> >
                    (thrust::multiplies<T>()));

  // Get optimal value
  _optval = FuncEval(f_gpu, y12.data);
  MPI_Allreduce(MPI_IN_PLACE, &_optval, 1, t_type, MPI_SUM, MPI_COMM_WORLD);
  _optval = _optval + FuncEval(g_gpu, x12.data);

  // Check status
  PogsStatus status;
  if (!converged && k == _max_iter - 1)
    status = POGS_MAX_ITER;
  else if (!converged && k < _max_iter - 1)
    status = POGS_NAN_FOUND;
  else
    status = POGS_SUCCESS;

  // Print summary
  MASTER(kRank) {
    if (_verbose > 0) {
      double total = timer<double>() - t0;
      Printf(__HBAR__
             "Status: %s\n"
             "Timing: Total = %3.2e s, Init = %3.2e s\n"
             "Iter  : %u\n",
             PogsStatusString(status).c_str(), timer<double>() - t0, time_init, k);
      Printf(__HBAR__
             "Error Metrics:\n"
             "Pri: "
             "|Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = %.2e\n"
             "Dua: "
             "|A'l + u|   / (abs_tol sqrt(n)     / rels, Init = %3.2e s\n"
             "Iter  : %u\n",
             PogsStatusString(status).c_str(), timer<double>() - t0, time_init, k);
      Printf(__HBAR__
             "Error Metrics:\n"
             "Pri: "
             "|Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = %.2e\n"
             "Dua: "
             "|A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = %.2e\n"
             "Gap: "
             "|x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = %.2e\n"
             __HBAR__, _rel_tol * nrm_r / eps_pri, _rel_tol * nrm_s / eps_dua,
             _rel_tol * gap / eps_gap);

      BMARK_PRINT_T("final_optval", _optval);
      BMARK_PRINT_T("total_time", total);
      BMARK_PRINTF("iterations", "%d", k);
    }
  }

  // Scale x, y, lambda and mu for output.
  cml::vector_memcpy(&ztemp, &zt);
  cml::blas_axpy(hdl, -kOne, &zprev, &ztemp);
  cml::blas_axpy(hdl, kOne, &z12, &ztemp);
  cml::blas_scal(hdl, -_rho, &ztemp);
  cml::vector_mul(&ytemp, &d);
  cml::vector_div(&xtemp, &e);

  cml::vector_div(&y12, &d);
  cml::vector_mul(&x12, &e);

  // Copy results to output.
  cml::vector_memcpy(_x, &x12);
  cml::vector_memcpy(_y, &y12);
  cml::vector_memcpy(_mu, &xtemp);
  cml::vector_memcpy(_lambda, &ytemp);

  // Store z.
  cml::vector_memcpy(&z, &zprev);

  // Free memory.
  cml::vector_free(&z12);
  cml::vector_free(&zprev);
  cml::vector_free(&ztemp);
  cml::vector_free(&x_avg);
  cml::vector_free(&x_avg_temp);
  cublasDestroy(hdl);
  CUDA_CHECK_ERR();

  MPI_Comm_free(&avg_comm);
  MPI_Comm_free(&exch_comm);

  return status;
}

template <typename T, typename M, typename P>
Pogs<T, M, P>::~Pogs() {
  cudaFree(_de);
  cudaFree(_z);
  cudaFree(_zt);
  _de = _z = _zt = 0;
  CUDA_CHECK_ERR();

  delete [] _x;
  delete [] _y;
  delete [] _mu;
  delete [] _lambda;
  _x = _y = _mu = _lambda = 0;
}

// Explicit template instantiation.
// Dense direct.
template class Pogs<double, MatrixDistDense<double>,
    ProjectorDirect<double, MatrixDistDense<double> > >;
template class Pogs<float, MatrixDistDense<float>,
    ProjectorDirect<float, MatrixDistDense<float> > >;

// Dense indirect.
// template class Pogs<double, MatrixDistDense<double>,
//     ProjectorCgls<double, MatrixDistDense<double> > >;
// template class Pogs<float, MatrixDistDense<float>,
//     ProjectorCgls<float, MatrixDistDense<float> > >;

// Sparse indirect.
// template class Pogs<double, MatrixSparse<double>,
//     ProjectorCgls<double, MatrixSparse<double> > >;
// template class Pogs<float, MatrixSparse<float>,
//     ProjectorCgls<float, MatrixSparse<float> > >;

}  // namespace pogs
