bash make_squad.sh
export SQUAD_DIR=squad1
python partition_squad_models.py --partitioning_batch_size 4 --bwd_to_fwd_ratio 5 --n_iter 50   --model_type bert   --model_name_or_path bert-base-uncased  --do_lower_case   --train_file $SQUAD_DIR/train-v1.1.json   --predict_file $SQUAD_DIR/dev-v1.1.json    --max_seq_length 384   --doc_stride 128 

# FIXME: Fails.