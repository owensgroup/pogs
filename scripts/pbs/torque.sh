#! /bin/bash
#PBS -S /bin/bash
#PBS -N testjob
#PBS -o testjob.out
#PBS -e testjob.err
#PBS -M abpoms@ucdavis.edu
#PBS -m abe

test_file=$1

module load boost/1.57
module load cuda/6.5
module load gcc/4.8.4
module load hwloc/1.7.1
module load numactl/2.0.9
module load openmpi-1.8.4/gcc-4.8.4/cuda-6.5
module load python/2.7.4

cd ~/repos/pogs
source .env/bin/activate
cd ~/repos/pogs/scripts
python benchmark.py pbs/${test_file} pbs/results/${test_file}_results.json
deactivate
