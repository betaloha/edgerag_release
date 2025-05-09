# Run with cache
#!/bin/bash
max_probe_cost=60000
num_query=10
mkdir -p results/perf/
for dataset in `seq 0 5` ; do
    for config in 6 ; do
        python3 test_driver.py --workload_idx $dataset --config_idx $config --num_query $num_query --load_cache --save_cache_checkpoint --max_probe_cost $max_probe_cost --debug
        python3 test_driver.py --workload_idx $dataset --config_idx $config --num_query $num_query --num_skip 10 --save_cache_checkpoint --load_cache_checkpoint --max_probe_cost $max_probe_cost --debug
    done
done
