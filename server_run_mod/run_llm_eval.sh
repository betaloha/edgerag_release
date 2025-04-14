#!/bin/bash

num_query=100
mkdir -p results/llm_eval/
for dataset in `seq 0 5` ; do
    for config in 0 3 ; do
        python3 test_driver.py --llm_evaluation --workload_idx $dataset --config_idx $config --num_query $num_query >> results/llm_eval/$dataset"_"$config".txt"
    done
done
