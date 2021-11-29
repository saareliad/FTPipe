mkdir /home_local/saareliad/data/moved_cache
#export OMP_NUM_THREADS=5
ulimit -n 16384
# export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d # Start deamon in background

### GPIPE MPIPE

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
echo quit | nvidia-cuda-mps-control
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe_op_graph.json --seed 42 --mode preproc
nvidia-cuda-mps-control -d # Start deamon in background
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe_op_graph.json --seed 42 --epochs_from_cmd --epochs 10 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 16 --bs_train_from_cmd --bs_train 8
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe_op_graph.json --seed 42 --mode eval

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
echo quit | nvidia-cuda-mps-control
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe_layer_graph.json --seed 42 --mode preproc
nvidia-cuda-mps-control -d # Start deamon in background
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe_layer_graph.json --seed 42 --epochs_from_cmd --epochs 10 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 16 --bs_train_from_cmd --bs_train 8
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe_layer_graph.json --seed 42 --mode eval

#### FTPIPE MPIPE
mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
echo quit | nvidia-cuda-mps-control
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_op_graph.json --seed 42 --mode preproc
nvidia-cuda-mps-control -d # Start deamon in background
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_op_graph.json --seed 42 --epochs_from_cmd --epochs 10 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 4 --bs_train_from_cmd --bs_train 32
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_op_graph.json --seed 42 --mode eval

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --epochs_from_cmd --epochs 10 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 4 --bs_train_from_cmd --bs_train 32
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode eval

#### STALE SEQ
echo quit | nvidia-cuda-mps-control

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq_op_graph/wic/pipedream_stale.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq_op_graph/wic/pipedream_stale.json --seed 42 --epochs_from_cmd --epochs 10 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 8 --bs_train_from_cmd --bs_train 16
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq_op_graph/wic/pipedream_stale.json --seed 42 --mode eval

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42 --epochs_from_cmd --epochs 10 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 8 --bs_train_from_cmd --bs_train 16
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42 --mode eval

###### GPIPE SEQ

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq_op_graph/wic/gpipe_new.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq_op_graph/wic/gpipe_new.json --seed 42 --epochs_from_cmd --epochs 10 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 16 --bs_train_from_cmd --bs_train 8
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq_op_graph/wic/gpipe_new.json --seed 42 --mode eval

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/gpipe_new.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/gpipe_new.json --seed 42 --epochs_from_cmd --epochs 5 --steps_from_cmd --steps -1 --step_every_from_cmd --step_every 16 --bs_train_from_cmd --bs_train 8
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/gpipe_new.json --seed 42 --mode eval
