# Prefetch script for cache
#!/bin/bash
max_probe_cost=-1
num_query=-1
min_cache_cost=0
mkdir -p results/prefetch/cost_$max_probe_cost/
for dataset in `seq 0 5` ; do
    for config in 6 ; do
        python3 test_driver.py --workload_idx $dataset --config_idx $config --num_query $num_query --prefetch --max_probe_cost $max_probe_cost --min_cache_cost $min_cache_cost  >> results/prefetch/cost_$max_probe_cost/$dataset
    done
done
