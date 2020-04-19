from run.sequential_sim_set import run_grid_on_multi_gpu_per_run
import os
import subprocess


def find_best(path):
    files = os.listdir(path)
    for f in files:
        if not f.endswith("py"):
            continue
        basename = os.path.basename(f)

        print(
            basename, ": ",
            subprocess.check_output(['tail', '-2',
                                     os.path.join(path, f)]).decode())


def gpt2xl_tied_p8():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    param_grid = dict(
        seed=[42],
        model_type=["gpt2"],
        model_name_or_path=["gpt2-xl"],
        train_data_file=[TRAIN_FILE],
        n_partitions=[8],
        partitioning_batch_size=[1],
        analysis_batch_size=[1],
        block_size=[-1],
        n_iter=[1],
        lmhead=[""],
        async_pipeline=[""],
        auto_file_name=[""],
        overwrite_cache=[""],
        bwd_to_fwd_ratio=[1, 2, 3, 4, 5, 6, -1],
        # bwd_to_fwd_ratio=[-1],
        output_file=["results/gpt2xl_tied_p8/"],
        stateless_tied=[""])
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)
    find_best("results/gpt2xl_tied_p8/")


def gpt2xl_untied_p8():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    param_grid = dict(
        seed=[42],
        model_type=["gpt2"],
        model_name_or_path=["gpt2-xl"],
        train_data_file=[TRAIN_FILE],
        n_partitions=[8],
        partitioning_batch_size=[1],
        analysis_batch_size=[1],
        block_size=[-1],
        n_iter=[1, 10, 30, 100],
        lmhead=[""],
        async_pipeline=[""],
        auto_file_name=[""],
        overwrite_cache=[""],
        bwd_to_fwd_ratio=[1, 2, 3, 4, 5, 6, -1],
        # bwd_to_fwd_ratio=[-1],
        output_file=["results/gpt2xl_untied_p8/"],
    )
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)
    find_best("results/gpt2xl_untied_p8/")


def gpt2_tied_p4():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    OUT_DIR = "results/new/new_40_iter/"
    param_grid = dict(
        seed=[42],
        model_type=["gpt2"],
        model_name_or_path=["gpt2"],
        train_data_file=[TRAIN_FILE],
        n_partitions=[4],
        partitioning_batch_size=[4],
        analysis_batch_size=[4],
        block_size=[-1],
        n_iter=[40],
        lmhead=[""],
        async_pipeline=[""],
        auto_file_name=[""],
        overwrite_cache=[""],
        bwd_to_fwd_ratio=[1, 2, 3, 4, 5, 6, -1],
        # bwd_to_fwd_ratio=[-1],
        # output_file=["models/partitioned/tmp/"],
        output_file=[OUT_DIR],
        stateless_tied=[""])
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)

    find_best(OUT_DIR)


if __name__ == "__main__":
    # gpt2_tied_p4()
    gpt2xl_tied_p8()
    gpt2xl_untied_p8()
    # gpt2xl_tied_p8()
