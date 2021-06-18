export OMP_NUM_THREADS=5

# First, eval a seqpipe run
#python -m pipe.main --config pipe/configs/t5/t5_3b_p8/seq/boolq/gpipe_new.json --seed 42 --mode eval

mkdir /home_local/saareliad/data/moved_cache
# Now, run mpipe

# opgraph

mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
echo quit | nvidia-cuda-mps-control
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/gpipe_op_graph.json --seed 42 --mode preproc
nvidia-cuda-mps-control -d # Start deamon in background
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/gpipe_op_graph.json --seed 42
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe_op_graph.json --seed 42 --mode eval

# layers graph
# boolq
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42 --mode eval

# multirc
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42 --mode eval

# wic
bash prepare_new_t5/to_partition_spipe_t5_3b_wic.sh
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode eval

# rte
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42 --mode eval

# rte
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/gpipe_layer_graph.json --seed 42 --mode preproc
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/gpipe_layer_graph.json --seed 42
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/gpipe_layer_graph.json --seed 42 --mode eval
echo OK
# seqpipe layergraph

# wic
mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42
python -m pipe.main --config pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42 --mode eval

# boolq
bash prepare_new_t5/to_partition_spipe_t5_3b_boolq_multirc.sh
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/boolq/pipedream_stale.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/boolq/pipedream_stale.json --seed 42
python -m pipe.main --config pipe.main --config pipe/configs/t5/new_t5_exp/seq/boolq/pipedream_stale.json --seed 42 --mode eval

mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/boolq/gpipe_new.json --seed 42


# multirc
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/multirc/pipedream_stale.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/multirc/pipedream_stale.json --seed 42
python -m pipe.main --config pipe.main --config pipe/configs/t5/new_t5_exp/seq/multirc/pipedream_stale.json --seed 42 --mode eval

bash prepare_new_t5/to_partition_spipe_t5_3b_rte.sh
mv /home_local/saareliad/data/cache_* /home_local/saareliad/data/moved_cache/
python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/rte/pipedream_stale.json --seed 42 --mode preproc
mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/rte/pipedream_stale.json --seed 42
python -m pipe.main --config pipe.main --config pipe/configs/t5/new_t5_exp/seq/rte/pipedream_stale.json --seed 42 --mode eval

################## Preproc #####################
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42 --mode preproc
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42 --mode preproc
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42 --mode preproc
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode preproc
#
################## Run #####################
#
#
#mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42
#mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42
#mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42
#mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42
#
#
################## Eval #####################
#
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42 --mode eval
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42 --mode eval
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42 --mode eval
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode eval
#
#
################ GPipe ( no eval ) ######################

# mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/gpipe.json --seed 42
# mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/gpipe.json --seed 42
#mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/gpipe.json --seed 42
#mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/gpipe.json --seed 42

############### partition the rest ######################

# Then: TODO: check if transformer_cfg exists, else add
# Then: TODO: manualy add stage to device map from generated files to configs
# Then: TODO: manualy add configs for mpipe layers graph
