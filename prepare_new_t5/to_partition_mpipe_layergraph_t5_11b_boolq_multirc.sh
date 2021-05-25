rm new_trace_cache_t511b_512_4_lg new_prof_cache_t511b_512_4_lg_ftpipe
python -m autopipe.partition new_t5 \
 --model_name_or_path \
 t5-11b \
 --t5_task \
 squad1 \
 --lmhead \
 --n_iter \
 2 \
 --analysis_batch_size \
 1 \
 --partitioning_batch_size \
 1 \
 --ct \
 new_trace_cache_t511b_512_4_lg \
 --cp \
 new_prof_cache_t511b_512_4_lg_ftpipe \
 --stateless_tied \
 --lmhead \
 --n_partitions \
 4 \
 --L \
 8 \
 16 \
 --max_seq_length \
 512 \
 --answer_max_seq_length \
 4 \
 --partitioning_method \
  mpipe \
 --preset \
  ftpipe \
 --dont_use_async_meta_alg \
 --save_memory_mode \
 --disable_op_profiling \
 --special_blocks \
 T5Block \
 --basic_blocks \
 T5Block \
 --output_file \
 layer_graph_ 
 # > partitioning_output_mpipe_t53b_512_4_lg_ftpipe.txt

