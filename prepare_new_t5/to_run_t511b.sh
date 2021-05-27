export OMP_NUM_THREADS=2

#mkdir /home_local/saareliad/data/moved_cache
nvidia-cuda-mps-control -d # Start deamon in background
python -m pipe.main --config pipe/configs/t5/t511b/boolq/gpipe_layer_graph.json --seed 42 --mode preproc
mpirun -np 7 python -m pipe.main --config pipe/configs/t5/t511b/boolq/gpipe_layer_graph.json --seed 42

exit 0
nvidia-cuda-mps-control -d # Start deamon in background
python -m pipe.main --config pipe/configs/t5/t511b/boolq/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 7 python -m pipe.main --config pipe/configs/t5/t511b/boolq/stale_layer_graph.json --seed 42


#### spipe
# export OMP_NUM_THREADS=4
# bash prepare_new_t5/to_partition_spipe_layergraph_t5_11b_boolq_multirc.sh 
# python -m pipe.main --config pipe/configs/t5/t511b/boolq/stale_layer_graph_pipedream.json --seed 42 --mode preproc
# mpirun -np 4 python -m pipe.main --config pipe/configs/t5/t511b/boolq/stale_layer_graph_pipedream.json --seed 42
