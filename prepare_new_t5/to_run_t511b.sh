export OMP_NUM_THREADS=2
mkdir /home_local/saareliad/data/moved_cache
nvidia-cuda-mps-control -d # Start deamon in background

python -m pipe.main --config pipe/configs/t5/t511b/boolq/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 7 python -m pipe.main --config pipe/configs/t5/t511b/boolq/stale_layer_graph.json --seed 42

