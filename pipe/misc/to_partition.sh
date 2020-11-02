### Partitioning without virtual stages
# boolq
python partitioning_script.py t5 --model_name_or_path t5-3b --t5_task squad1 --lmhead --n_iter 10 --analysis_batch_size 4 --partitioning_batch_size 4 --precompute_masks --stateless_tied --lmhead --n_partitions 8 --max_seq_length 512 --answer_max_seq_length 4 --basic_blocks T5Block --save_memory_mode --partitioning_method ACYCLIC --constraint memory --objective stage_time --multilevel

# wic
python partitioning_script.py t5 --model_name_or_path t5-3b --t5_task squad1 --lmhead --n_iter 10 --analysis_batch_size 32 --partitioning_batch_size 32 --precompute_masks --stateless_tied --lmhead --n_partitions 8 --max_seq_length 64 --answer_max_seq_length 4 --basic_blocks T5Block --save_memory_mode --partitioning_method ACYCLIC --constraint memory --objective stage_time --multilevel

# rte (could be 317,6, but I rounded)
python partitioning_script.py t5 --model_name_or_path t5-3b --t5_task squad1 --lmhead --n_iter 10 --analysis_batch_size 4 --partitioning_batch_size 4 --precompute_masks --stateless_tied --lmhead --n_partitions 8 --max_seq_length 320 --answer_max_seq_length 8 --basic_blocks T5Block --save_memory_mode --partitioning_method ACYCLIC --constraint memory --objective stage_time --multilevel

###

# bert squad (new version), TODO: metis...
python partitioning_script.py bert_squad --n_partitions 8 --partitioning_batch_size 24 --bwd_to_fwd_ratio 1 --n_iter 10 --model_name_or_path bert-large-uncased --do_lower_case --train_file /home_local/saareliad/data/squad1/train-v1.1.json --max_seq_length 384 --doc_stride 128 --async_pipeline --basic_blocks BertSelfAttention