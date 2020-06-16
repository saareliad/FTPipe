# TODO: add all new options...
bash make_squad.sh
export SQUAD_DIR=squad1
python partition_squad_models.py --n_partitions 8 --partitioning_batch_size 24 --analysis_batch_size 24 --bwd_to_fwd_ratio 3 --n_iter 50 --auto_file_name --model_type bert --model_name_or_path bert-large-uncased-whole-word-masking --do_lower_case --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json --max_seq_length 384 --doc_stride 128 --async_pipeline
