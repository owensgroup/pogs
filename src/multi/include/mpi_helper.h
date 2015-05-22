#ifndef MPI_HELPER_H_
#define MPI_HELPER_H_

#include <mpi.h>
#include <vector>
#include "cml/cml_blas.cuh"

#define MASTER(rank) if (rank == 0)

#define MVAPICH false

namespace mpih {

namespace {
  template<class T, class U>
    struct is_same {
      enum { value = 0 };
    };

  template<class T>
    struct is_same<T, T> {
    enum { value = 1 };
  };
}

template <typename T>
inline MPI_Datatype MPIDTypeFromT() {
  return is_same<T,double>::value ? MPI_DOUBLE : MPI_FLOAT;
}

template <typename T>
T dist_blas_dot(cublasHandle_t handle,
                const cml::vector<T> *x, const cml::vector<T> *y) {
  T norm2;
  cml::blas_dot(handle, x, y, &norm2);
  MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, MPIDTypeFromT<T>(), MPI_SUM,
                MPI_COMM_WORLD);
  return norm2;
}

template <typename T>
T dist_blas_nrm2(cublasHandle_t handle, const cml::vector<T> *x) {
  return sqrtf(dist_blas_dot(handle, x, x));
}

template <typename T>
int Allreduce(cublasHandle_t b_hdl,
              T *send, T *recv, int count, MPI_Op op, MPI_Comm comm) {
  // MVAPICH
#if MVAPICH
  return MPI_Allreduce(send, recv, count, op, comm);
#else
  // OpenMPI
  if (op != MPI_SUM) {
    printf("Allreduce only supports MPI_SUM right now\n");
    exit(-1);
  }
  int commSize;
  MPI_Comm_size(comm, &commSize);

  MPI_Datatype t_type = MPIDTypeFromT<T>();

  cml::vector<T> recv = cml::vector_view(recv, count);

  cml::matrix<T, CblasRowMajor> gather_buf =
    cml::matrix_calloc<T, CblasRowMajor>(commSize, count);
  cml::vector<T> ident = cml::vector_alloc<T>(commSize);

  cml::vector_set_all(&ident, kOne);

  MPI_Allgather(send, count, t_type, gather_buf.data, count, t_type, comm);
  cml::blas_gemv(b_hdl, CUBLAS_OP_T, kOne, &gather_buf, &ident, 0, recv);

  cml::matrix_free(&gather_buf);
  cml::vector_free(&ident);
}

}

#endif // MPI_HELPER_H_
