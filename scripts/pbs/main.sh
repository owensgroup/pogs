#! /bin/bash

test_files=(lasso2 lasso4 lasso8 lasso16 lasso32)
test_files=("${test_files[@]/%/.json}" )

for i in ${test_files[@]}
do
    qsub -F "$i" torque.sh
done

