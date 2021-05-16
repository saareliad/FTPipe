
rm new_trace_cache_t53b_*
rm new_prof_cache_t53b_*
rm prof_cache_t53b_64_4_lg_ftpipe
rm new_prof_cache_t53b_64_4_lg_ftpipe
rm new_trace_cache_t53b_64_4_lg

### mpipe (layers graph)
# bash prepare_new_t5/to_partition_mpipe_layergraph_t5_3b_boolq_multirc.sh
# bash prepare_new_t5/to_partition_mpipe_layergraph_t5_3b_rte.sh
bash prepare_new_t5/to_partition_mpipe_layergraph_t5_3b_wic.sh  #Failed to size?

### mpipe (op graph)
#bash prepare_new_t5/to_partition_mpipe_t5_3b_opgraph_boolq_multirc.sh  # FAILED: MEM
#bash prepare_new_t5/to_partition_mpipe_t5_3b_opgraph_rte.sh
bash prepare_new_t5/to_partition_mpipe_t5_3b_opgraph_wic.sh


### spipe (layers graph)
#bash prepare_new_t5/to_partition_spipe_t5_3b_boolq_multirc.sh
#bash prepare_new_t5/to_partition_spipe_t5_3b_rte.sh
bash prepare_new_t5/to_partition_spipe_t5_3b_wic.sh


### spipe (op graph)
bash prepare_new_t5/to_partition_spipe_OP_t5_3b_wic.sh

# TODO: gpipe: partition with smaller micro batch.

#to_partition_mpipe_t5_base
#to_partition_spipe_t5
#to_partition_spipe_t5_base