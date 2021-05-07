python -m autopipe.partition new_t5 \
 --model_name_or_path \
 t5-3b \
 --t5_task \
 squad1 \
 --lmhead \
 --n_iter \
 10 \
 --analysis_batch_size \
 8 \
 --partitioning_batch_size \
 8 \
 --ct \
 new_trace_cache_t53b_320_8_lg \
 --cp \
 new_prof_cache_t53b_320_8_lg \
 --stateless_tied \
 --lmhead \
 --n_partitions \
 8 \
 --max_seq_length \
 320 \
 --answer_max_seq_length \
 8 \
 --partitioning_method \
  pipedream \
 --preset \
  pipedream \
  --disable_op_profiling \
 --dont_use_async_meta_alg \
 --save_memory_mode \
 --special_blocks \
 T5Block \
 --basic_blocks \
 T5Block \
 --output_file \
  layer_graph_

# --basic_blocks \
# T5Block

