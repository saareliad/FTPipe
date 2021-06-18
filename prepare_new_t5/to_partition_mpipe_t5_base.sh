python -m autopipe.partition new_t5 \
 --model_name_or_path \
 t5-base \
 --t5_task \
 squad1 \
 --lmhead \
 --n_iter \
 1 \
 --analysis_batch_size \
 2 \
 --partitioning_batch_size \
 2 \
 --stateless_tied \
 --lmhead \
 --n_partitions \
 4 \
 --L \
 4 \
 8 \
 12 \
 16 \
 --max_seq_length \
 512 \
 --answer_max_seq_length \
 4 \
 --partitioning_method \
  mpipe \
 --save_memory_mode \
 --output_file \
 lg
 --special_blocks \
 T5Block \
 --basic_blocks \
 T5Block

