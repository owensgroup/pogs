#ifndef MPI_UTIL_H_
#define MPI_UTIL_H_

#include <mpi.h>

namespace mpiu {
///////////////////////////////////////////////////////////////////////////////
/// MPI functions specialized for type
///////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////// 
// Allreduce

inline int Allreduce(float *send,
                     float *recv,
                     int count,
                     MPI_Op op,
                     MPI_Comm comm) {
  return MPI_Allreduce(send, recv, count, MPI_FLOAT, op, comm);
}

inline int Allreduce(double *send,
                     double *recv,
                     int count,
                     MPI_Op op,
                     MPI_Comm comm) {
  return MPI_Allreduce(send, recv, count, MPI_DOUBLE, op, comm);
}

///////////////////////////////////////////////////////////////////////////////
// Allgather

inline int Allgather(float *send,
                     int send_count,
                     float *recv,
                     int recv_count,
                     MPI_Comm comm) {
  return MPI_Allgather(send, send_count, MPI_FLOAT, recv, recv_count, MPI_FLOAT,
                       comm);
};

inline int Allgather(double *send,
                     int send_count,
                     double *recv,
                     int recv_count,
                     MPI_Comm comm) {
  return MPI_Allgather(send, send_count, MPI_DOUBLE, recv, recv_count,
                       MPI_DOUBLE, comm);
};

///////////////////////////////////////////////////////////////////////////////
// Gather

inline int Gather(float *send,
                  int send_count,
                  float *recv,
                  int recv_count,
                  int root,
                  MPI_Comm comm) {
  return MPI_Gather(send, send_count, MPI_FLOAT, recv, recv_count, MPI_FLOAT,
                    root, comm);
};

inline int Gather(double *send,
                  int send_count,
                  double *recv,
                  int recv_count,
                  int root,
                  MPI_Comm comm) {
  return MPI_Gather(send, send_count, MPI_DOUBLE, recv, recv_count, MPI_DOUBLE,
                    root, comm);
};

inline int Scatter(float *send,
                  int send_count,
                  float *recv,
                  int recv_count,
                  int root,
                  MPI_Comm comm) {
  return MPI_Scatter(send, send_count, MPI_FLOAT, recv, recv_count, MPI_FLOAT,
                    root, comm);
};

inline int Scatter(double *send,
                  int send_count,
                  double *recv,
                  int recv_count,
                  int root,
                  MPI_Comm comm) {
  return MPI_Scatter(send, send_count, MPI_DOUBLE, recv, recv_count, MPI_DOUBLE,
                    root, comm);
};

inline int Bcast(double *send,
                 int send_count,
                 int root,
                 MPI_Comm comm) {
  return MPI_Bcast(send, send_count, MPI_DOUBLE, root, comm);
};

inline int Bcast(float *send,
                  int send_count,
                  int root,
                  MPI_Comm comm) {
  return MPI_Bcast(send, send_count, MPI_FLOAT, root, comm);
};

} // namespace mpiu
#endif // MPI_UTIL_H
