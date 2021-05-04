
export OMP_NUM_THREADS=5

# First, eval a seqpipe run
#python -m pipe.main --config pipe/configs/t5/t5_3b_p8/seq/boolq/gpipe_new.json --seed 42 --mode eval


# Now, run mpipe

# layers graph
# boolq
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/boolq/stale_layer_graph.json --seed 42 --mode eval

# multirc
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 15 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42
#python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/multirc/stale_layer_graph.json --seed 42 --mode eval

# wic
# python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode preproc
# mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/wic/stale_layer_graph.json --seed 42 --mode eval

# rte
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42 --mode preproc
mpirun -np 16 python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42
python -m pipe.main --config pipe/configs/t5/new_t5_exp/mpipe/rte/stale_layer_graph.json --seed 42 --mode eval

# seqpipe layergraph

 # wic
 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42 --mode preproc
 mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42
 python -m pipe.main --config pipe.main --config pipe/configs/t5/new_t5_exp/seq/wic/pipedream_stale.json --seed 42 --mode eval

# boolq
 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/boolq/pipedream_stale.json --seed 42 --mode preproc
 mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/boolq/pipedream_stale.json --seed 42
 python -m pipe.main --config pipe.main --config pipe/configs/t5/new_t5_exp/seq/boolq/pipedream_stale.json --seed 42 --mode eval

# multirc
 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/multirc/pipedream_stale.json --seed 42 --mode preproc
 mpirun -np 8 python -m pipe.main --config pipe/configs/t5/new_t5_exp/seq/multirc/pipedream_stale.json --seed 42
 python -m pipe.main --config pipe.main --config pipe/configs/t5/new_t5_exp/seq/multirc/pipedream_stale.json --seed 42 --mode eval




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
