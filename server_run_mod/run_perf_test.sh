#!/bin/bash
num_query=100
num_run=1
# Use Default probe cost.
max_probe_cost=-1
mkdir -p results/perf/
for dataset in `seq 0 5` ; do
    for config in 0 3 4 5 ; do
        for itr in `seq 0 $((num_run-1))` ; do
            offset=$((itr*num_query))
            python3 test_driver.py --workload_idx $dataset --config_idx $config --num_query $num_query --num_skip $offset --max_probe_cost $max_probe_cost >> results/perf/$dataset"_"$config".txt"
        done
    done
done
