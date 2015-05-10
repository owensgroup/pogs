#ifndef MPI_HELPER_H_
#define MPI_HELPER_H_

#include <mpi.h>
#include <vector>
#include "cml/cml_blas.cuh"

#define MASTER(rank) if (rank == 0)

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

}

#endif // MPI_HELPER_H_
