from run.sequential_sim_set import run_grid_on_multi_gpu_per_run

"""export TRAIN_FILE=wikitext-2-raw/wiki.train.raw
python partition_gpt2_models.py \
    --model_type gpt2 \
    --model_name_or_path gpt2 \
    --train_data_file $TRAIN_FILE \
    --n_partitions 4 \
    --partitioning_batch_size 4 \
    --analysis_batch_size 4 \
    --block_size -1 \
    --n_iter 20 \
    --lmhead \
    --async_pipeline \
    --output_file gpt2_tied_lm_5p \
    --overwrite_cache \
    --bwd_to_fwd_ratio 5
"""


def main():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE="wikitext-2-raw/wiki.train.raw"
    param_grid = dict( seed=[42],
    model_type=["gpt2"],
    model_name_or_path=["gpt2"],
    train_data_file=[TRAIN_FILE],
    n_partitions=[4],
    partitioning_batch_size=[4],
    analysis_batch_size=[4],
    block_size=[-1],
    n_iter=[1],
    lmhead=[""],
    async_pipeline=[""],
    auto_file_name=[""],
    # output_file=["gpt2_tied_lm_5p"],
    overwrite_cache=[""],
    bwd_to_fwd_ratio=[1,2,3,4,5,6,-1],
    stateless_tied=[""]
    )
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)
if __name__ == "__main__":
    main()
