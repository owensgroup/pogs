#! /bin/bash
#PBS -S /bin/bash
#PBS -M abpoms@ucdavis.edu
#PBS -m abe

result_file=$1
test_arg=$2


cd ~/repos/pogs
source .env/bin/activate
cd ~/repos/pogs/scripts/benchmark
source modules_mvapich_gdr.sh
python benchmark.py --spec ../ref_spec.json --plan ${test_arg} --results results/${result_file}_results.json
deactivate
