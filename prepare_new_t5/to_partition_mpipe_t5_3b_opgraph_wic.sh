rm new_prof_cache_t53b_64_4_op_ftpipe new_trace_cache_t53b_64_4_op
python -m autopipe.partition new_t5 \
 --model_name_or_path \
 t5-3b \
 --t5_task \
 squad1 \
 --lmhead \
 --n_iter \
 10 \
 --analysis_batch_size \
 32 \
 --partitioning_batch_size \
 32 \
 --ct \
 new_trace_cache_t53b_64_4_op \
 --cp \
 new_prof_cache_t53b_64_4_op_ftpipe \
 --stateless_tied \
 --lmhead \
 --n_partitions \
 8 \
 --L \
 16 \
 --max_seq_length \
 64 \
 --answer_max_seq_length \
 4 \
 --partitioning_method \
  mpipe \
 --preset \
  ftpipe \
 --dont_use_async_meta_alg \
 --save_memory_mode \
 --special_blocks \
 T5Block \
 --output_file \
  op_

# --output_file \
#  lg \
# --basic_blocks \
# T5Block

