#!/bin/bash

for dataset in `seq 0 5`; do
    for config in 0 3; do
        python3 beir_bench_single.py --workload_idx $dataset --config_idx $config >> beir_results.txt
    done
done