#! /bin/bash
#PBS -S /bin/bash
#PBS -M abpoms@ucdavis.edu
#PBS -m abe

cd ~/repos/pogs/scripts/benchmark
source modules_mvapich_gdr.sh

cd ~/repos/pogs
source .env/bin/activate

cd ~/repos/pogs/scripts/benchmark
python benchmark.py "$@"
deactivate
