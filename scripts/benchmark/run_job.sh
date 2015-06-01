#! /bin/bash
#PBS -S /bin/bash
#PBS -M abpoms@ucdavis.edu
#PBS -m abe

result_file=$1
test_arg=$2

module load boost/1.57
module load cuda/7.0
module load gcc/4.8.4
module load hwloc/1.7.1
module load numactl/2.0.9
module load mvapich2-2.1rc2/gcc-4.8.4/cuda-7.0
module load python/2.7.4

export MV2_USE_CUDA=1

cd ~/repos/pogs
source .env/bin/activate
cd ~/repos/pogs/scripts/benchmark
python benchmark.py --spec ref_spec.json --plan ${test_arg} --results results/${result_file}_results.json
deactivate
