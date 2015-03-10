#include <cusparse.h>
#include <cublas_v2.h>

#include "sinkhorn_knopp.cuh"

#include <cmath>
#include <algorithm>
#include <vector>

#include "_interface_defs.h"
#include "cml/cml_linalg.cuh"
#include "cml/cml_spblas.cuh"
#include "cml/cml_spmat.cuh"
#include "cml/cml_vector.cuh"
#include "pogs.h"
#include "timer.hpp"

char test_sep = ':';

template <typename T>
inline void TestPrintT(const char *name, T value) {
  printf("%s %c %.3e\n", name, test_sep, value);
}

template <typename T>
inline void TestIterPrintT(unsigned int iter, const char *name, T value) {
  printf("iter, %d, %s %c %.3e\n", iter, name, test_sep, value);
}

template <typename T>
inline void TestIterPrintF(unsigned int iter, const char *name, T value) {
  printf("iter, %d, %s %c %.3f\n", iter, name, test_sep, value);
}

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

  double total_time, bcast_meta_time, allocate_aij_time, send_matrix_time,
    admm_allocate_time, preprocess_time, total_iter_time, total_prox_time,
    total_global_z_time, total_global_z12_time, total_proj_time,
    total_primal_time, total_avg_time, total_dual_time;

  total_iter_time = 0;
  total_prox_time = 0;
  total_global_z_time = 0;
  total_global_z12_time = 0;
  total_proj_time = 0;
  total_primal_time = 0;
  total_avg_time = 0;
  total_dual_time = 0;
  total_time = timer<double>();

  pogs_data->quiet = true;

  int err = 0;

  // Extract values from pogs_data
  int m = pogs_data->m, n = pogs_data->n, nnz = pogs_data->A.nnz;
  T rho = pogs_data->rho;
  thrust::device_vector<FunctionObj<T> > f = pogs_data->f;
  thrust::device_vector<FunctionObj<T> > g = pogs_data->g;
 
  // Create cuBLAS hdl.
  cublasHandle_t d_hdl;
  cublasCreate(&d_hdl);
  cusparseHandle_t s_hdl;
  cusparseCreate(&s_hdl);
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);

  // Allocate data for ADMM variables.
  admm_allocate_time = timer<double>();
  bool pre_process = true;
  cml::vector<T> de, z, zt;
  cml::vector<T> zprev = cml::vector_calloc<T>(m + n);
  cml::vector<T> z12 = cml::vector_calloc<T>(m + n);
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
    de = cml::vector_calloc<T>(m + n);
    z = cml::vector_calloc<T>(m + n);
    zt = cml::vector_calloc<T>(m + n);
    A = cml::spmat_alloc<T, typename M::I_t, kOrd>(m, n, nnz);
  }

  if (de.data == 0 || z.data == 0 || zt.data == 0 || zprev.data == 0 ||
      z12.data == 0 || A.val == 0 || A.ind == 0 || A.ptr == 0) {
    err = 1;
  }
  admm_allocate_time = timer<double>() - admm_allocate_time;

  // Create views for x and y components.
  cml::vector<T> d = cml::vector_subvector(&de, 0, m);
  cml::vector<T> e = cml::vector_subvector(&de, m, n);
  cml::vector<T> x = cml::vector_subvector(&z, 0, n);
  cml::vector<T> y = cml::vector_subvector(&z, n, m);
  cml::vector<T> x12 = cml::vector_subvector(&z12, 0, n);
  cml::vector<T> y12 = cml::vector_subvector(&z12, n, m);

  preprocess_time = timer<double>();
  if (pre_process && !err) {
    cml::spmat_memcpy(s_hdl, &A, pogs_data->A.val, pogs_data->A.ind,
        pogs_data->A.ptr);
    err = sinkhorn_knopp::Equilibrate(s_hdl, d_hdl, descr, &A, &d, &e);

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
  preprocess_time = timer<double>() - preprocess_time;

  // Scale f and g to account for diagonal scaling e and d.
  if (!err) {
    thrust::transform(f.begin(), f.end(), thrust::device_pointer_cast(d.data),
        f.begin(), ApplyOp<T, thrust::divides<T> >(thrust::divides<T>()));
    thrust::transform(g.begin(), g.end(), thrust::device_pointer_cast(e.data),
        g.begin(), ApplyOp<T, thrust::multiplies<T> >(thrust::multiplies<T>()));
  }

  // Signal start of execution.
  if (!pogs_data->quiet)
    Printf("   #      res_pri    eps_pri   res_dual   eps_dual"
           "        gap    eps_gap  objective\n");

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * pogs_data->abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * pogs_data->abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1.0);
  unsigned int kd = 0, ku = 0;
  bool converged = false;

  //double t = timer<double>();
  total_iter_time = timer<double>();
  unsigned int k;
  for (k = 0; !err; ++k) {
    double iter_time = timer<double>();
    double prox_time = 0;
    double global_z_time = 0;
    double global_z12_time = 0;
    double proj_time = 0;
    double primal_time = 0;
    double avg_time = 0;
    double dual_time = 0;
    cml::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    prox_time = timer<double>();
    cml::blas_axpy(d_hdl, -kOne, &zt, &z);
    ProxEval(g, rho, x.data, x.stride, x12.data, x12.stride);
    ProxEval(f, rho, y.data, y.stride, y12.data, y12.stride);
    prox_time = timer<double>() - prox_time;
    total_prox_time += prox_time;

    // Compute dual variable.
    T nrm_r = 0, nrm_s = 0, gap;
    cml::blas_axpy(d_hdl, -kOne, &z12, &z);
    cml::blas_dot(d_hdl, &z, &z12, &gap);
    gap = std::abs(gap);
    pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);
    T eps_gap = std::sqrt(static_cast<T>(m + n)) * pogs_data->abs_tol +
        pogs_data->rel_tol * cml::blas_nrm2(d_hdl, &z) *
        cml::blas_nrm2(d_hdl, &z12);
    T eps_pri = sqrtm_atol + pogs_data->rel_tol * cml::blas_nrm2(d_hdl, &z12);
    T eps_dua = sqrtn_atol + pogs_data->rel_tol * rho * 
        cml::blas_nrm2(d_hdl, &z);

    if (converged || k == pogs_data->max_iter)
      break;

    // Project and Update Dual Variables
    proj_time = timer<double>();
    cml::vector_memcpy(&y, &y12);
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, -kOne, &A,
        &x12, kOne, &y);
    primal_time = timer<double>();
    nrm_r = cml::blas_nrm2(d_hdl, &y);
    primal_time = timer<double>() - primal_time;
    total_primal_time += primal_time;
    cml::vector_set_all(&x, kZero);
    cml::spblas_solve(s_hdl, d_hdl, descr, &A, kOne, &y, &x, kTol, 5, true);
    cml::blas_axpy(d_hdl, kOne, &x12, &x);
    cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_NON_TRANSPOSE, descr, kOne, &A,
        &x, kZero, &y);
    proj_time = timer<double>() - proj_time;
    total_proj_time += proj_time;

    // Apply over relaxation.
    cml::blas_scal(d_hdl, kAlpha, &z);
    cml::blas_axpy(d_hdl, kOne - kAlpha, &zprev, &z);

    // Update dual variable.
    cml::blas_axpy(d_hdl, kAlpha, &z12, &zt);
    cml::blas_axpy(d_hdl, kOne - kAlpha, &zprev, &zt);
    cml::blas_axpy(d_hdl, -kOne, &z, &zt);

    dual_time = timer<double>();
    bool exact = false;
    cml::blas_axpy(d_hdl, -kOne, &zprev, &z12);
    cml::blas_axpy(d_hdl, -kOne, &z, &zprev);
    nrm_s = rho * cml::blas_nrm2(d_hdl, &zprev);
    if (nrm_r < eps_pri && nrm_s < eps_dua) {
      cml::spblas_gemv(s_hdl, CUSPARSE_OPERATION_TRANSPOSE, descr, kOne, &A,
          &y12, kOne, &x12);
      nrm_s = rho * cml::blas_nrm2(d_hdl, &x12);
      exact = true;
    }
    dual_time = timer<double>() - dual_time;
    total_dual_time += dual_time;

    // Evaluate stopping criteria.
    converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
        (!pogs_data->gap_stop || gap < eps_gap);
    if (!pogs_data->quiet && (k % 10 == 0 || converged))
      Printf("%4d :  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, pogs_data->optval);

    TestIterPrintF(k, "rho", rho);
    TestIterPrintF(k, "optval", pogs_data->optval);
    TestIterPrintF(k, "nrm_r", nrm_r);
    TestIterPrintF(k, "nrm_z", nrm_z);
    TestIterPrintF(k, "nrm_z12", nrm_z12);
    TestIterPrintF(k, "eps_pri", eps_pri);
    TestIterPrintF(k, "nrm_s", nrm_s);
    TestIterPrintF(k, "eps_dua", eps_dua);

    // Rescale rho.
    if (pogs_data->adaptive_rho && !converged) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        if (rho < kRhoMax) {
          rho *= delta;
          cml::blas_scal(d_hdl, 1 / delta, &zt);
          delta = kGamma * delta;
          ku = k;
          if (!pogs_data->quiet)
            Printf("+ rho %e\n", rho);
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (rho > kRhoMin) {
          rho /= delta;
          cml::blas_scal(d_hdl, delta, &zt);
          delta = kGamma * delta;
          kd = k;
          if (!pogs_data->quiet)
            Printf("- rho %e\n", rho);
        }
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = std::max(delta / kGamma, kDeltaMin);
      }
    }
  }
  total_iter_time = timer<double>() - total_iter_time;
  //Printf("TIME = %e\n", timer<double>() - t);

  // Scale x, y and l for output.
  cml::vector_div(&y12, &d);
  cml::vector_mul(&x12, &e);
  cml::vector_mul(&y, &d);
  cml::blas_scal(d_hdl, rho, &y);


  pogs_data->optval = FuncEval(f, y12.data, 1) + FuncEval(g, x12.data, 1);
    
  // Copy results to output.
  if (pogs_data->y != 0 && !err)
    cml::vector_memcpy(pogs_data->y, &y12);
  if (pogs_data->x != 0 && !err)
    cml::vector_memcpy(pogs_data->x, &x12);
  if (pogs_data->l != 0 && !err)
    cml::vector_memcpy(pogs_data->l, &y);

  if (pogs_data->y != 0 && !err) {
    T nrm = cml::blas_nrm2(d_hdl, &y12);
    TestPrintT("y_nrm", nrm);
  }
  if (pogs_data->x != 0 && !err) {
    T nrm = cml::blas_nrm2(d_hdl, &x12);
    TestPrintT("x_nrm", nrm);
  }
  if (pogs_data->l != 0 && !err) {
    T nrm = cml::blas_nrm2(d_hdl, &y);
    TestPrintT("l_nrm", nrm);
  }

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
  cml::vector_free(&z12);
  cml::vector_free(&zprev);
  
  TestPrintT("final_optval", pogs_data->optval);
  TestPrintT("total_time", timer<double>() - total_time);
  TestPrintT("bcast_meta_time", bcast_meta_time);
  TestPrintT("total_iter_time", total_iter_time);
  TestPrintT("send_matrix_time", send_matrix_time);
  TestPrintT("admm_allocate_time", admm_allocate_time);
  TestPrintT("preprocess_time", preprocess_time);
  TestPrintT("total_iter_time", total_iter_time);
  TestPrintT("total_prox_time", total_prox_time);
  TestPrintT("total_global_z_time", total_global_z_time);
  TestPrintT("total_global_z12_time", total_global_z12_time);
  TestPrintT("total_proj_time", total_proj_time);
  TestPrintT("total_primal_time", total_primal_time);
  TestPrintT("total_avg_time", total_avg_time);
  TestPrintT("total_dual_time", total_dual_time);
  printf("total_iterations %c %d\n", test_sep, k);


  return err;
}

template <typename T, typename I, POGS_ORD O>
int AllocSparseFactors(PogsData<T, Sparse<T, I, O> > *pogs_data) {
  size_t m = pogs_data->m, n = pogs_data->n, nnz = pogs_data->A.nnz;
  size_t flen = 1 + 3 * (n + m) + nnz;

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

