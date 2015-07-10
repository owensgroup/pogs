#! /bin/bash

module use /shared/apps/centos-6.6_SB/Bundles
module load PrgEnv/GCC+MVAPICH2/2015-06-17
module load mvapich2

module load boost/1.57
module load hwloc/1.7.1
module load numactl/2.0.9
module load python/2.7.4

export MV2_USE_CUDA=1
