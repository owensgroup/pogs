#! /bin/bash

test_files=(lasso2 lasso4 lasso8 lasso16 lasso32)
test_files=("${test_files[@]/%/.json}" )
test_config=(
    "nodes=1:k40x4:ppn=2:walltime=5:00:00"
    "nodes=1:k40x4:ppn=4:walltime=24:00:00"
    "nodes=2:k40x6:ppn=6:walltime=24:00:00"
    "nodes=3:k40x6:ppn=6:walltime=24:00:00"
    "nodes=6:k40x6:ppn=6:walltime=24:00:00"
)

for ((i=0;i<${#test_files[@]};++i)); do
    test=${test_files[i]}
    qsub -F "$test" -N $test -o ${test}.out -e ${test}.err -l ${test_config[i]} torque.sh
done
