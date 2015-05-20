#! /bin/bash

test_args=(single_d row_2_d row_4_d row_8_d row_16_d row_32_d)
out_files=("${test_args[@]/%/}" )
test_args=("${test_args[@]/%/:lasso_mn_d_1}" )
test_config=(
    "-l nodes=1:k40:ppn=1 -l walltime=24:00:00"
    "-l nodes=1:k40x4:ppn=2 -l walltime=24:00:00"
    "-l nodes=1:k40x4:ppn=4 -l walltime=24:00:00"
    "-l nodes=2:k40x6:ppn=6 -l walltime=24:00:00"
    "-l nodes=3:k40x6:ppn=6 -l walltime=24:00:00"
    "-l nodes=6:k40x6:ppn=6 -l walltime=24:00:00"
)

for ((i=0;i<${#test_args[@]};++i)); do
    test=${test_args[i]}
    out=${out_files[i]}
    qsub -F "${out_files[i]} $test" -N $out -o ${out}.out -e ${out}.err ${test_config[i]} torque.sh
done
