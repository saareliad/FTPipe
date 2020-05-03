from run.sequential_sim_set import run_grid_on_multi_gpu_per_run
import os
import subprocess

BWD_TO_FWD_RATIO_FULL_EXP = [1, 2, 3, 4, 5, -1]
BWD_TO_FWD_RATIO_BEST_TRANSFORMER = [2]
TRANSFORMER_USED_RATIO = BWD_TO_FWD_RATIO_FULL_EXP

N_ITER = [50]

# NOTE: values have to be in lists.


def find_best(path, do_print=True):
    files = os.listdir(path)
    res = dict()
    for f in files:
        if not f.endswith("py"):
            continue
        basename = os.path.basename(f)
        out = subprocess.check_output(['tail', '-2',
                                       os.path.join(path, f)]).decode()
        res[basename] = out
        if do_print:
            print(basename, ": ", out)
    # TODO: parse it better to get just the speedup.
    return res


def gpt2xl_tied_p8():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    OUT_DIR = "results/gpt2xl_tied_p8/"
    os.makedirs(OUT_DIR, exist_ok=True)

    param_grid = dict(seed=[42],
                      model_type=["gpt2"],
                      model_name_or_path=["gpt2-xl"],
                      train_data_file=[TRAIN_FILE],
                      n_partitions=[8],
                      partitioning_batch_size=[1],
                      analysis_batch_size=[1],
                      block_size=[-1],
                      n_iter=N_ITER,
                      lmhead=[""],
                      async_pipeline=[""],
                      auto_file_name=[""],
                      overwrite_cache=[""],
                      bwd_to_fwd_ratio=TRANSFORMER_USED_RATIO,
                      output_file=[OUT_DIR],
                      stateless_tied=[""])
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)
    find_best(OUT_DIR)


def gpt2xl_untied_p8():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    OUT_DIR = "results/gpt2xl_untied_p8/"
    os.makedirs(OUT_DIR, exist_ok=True)
    param_grid = dict(
        seed=[42],
        model_type=["gpt2"],
        model_name_or_path=["gpt2-xl"],
        train_data_file=[TRAIN_FILE],
        n_partitions=[8],
        partitioning_batch_size=[1],
        analysis_batch_size=[1],
        block_size=[-1],
        n_iter=N_ITER,
        lmhead=[""],
        async_pipeline=[""],
        auto_file_name=[""],
        overwrite_cache=[""],
        bwd_to_fwd_ratio=TRANSFORMER_USED_RATIO,
        output_file=[OUT_DIR],
    )
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)
    find_best(OUT_DIR)


def gpt2_tied_p4():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    OUT_DIR = "results/gpt2_tied_p4/"
    os.makedirs(OUT_DIR, exist_ok=True)

    param_grid = dict(seed=[42],
                      model_type=["gpt2"],
                      model_name_or_path=["gpt2"],
                      train_data_file=[TRAIN_FILE],
                      n_partitions=[4],
                      partitioning_batch_size=[4],
                      analysis_batch_size=[4],
                      block_size=[-1],
                      n_iter=N_ITER,
                      lmhead=[""],
                      async_pipeline=[""],
                      auto_file_name=[""],
                      overwrite_cache=[""],
                      bwd_to_fwd_ratio=TRANSFORMER_USED_RATIO,
                      output_file=[OUT_DIR],
                      stateless_tied=[""])
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)

    find_best(OUT_DIR)


def gpt2_untied_p4():
    COMMAND = "python partition_gpt2_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    OUT_DIR = "results/gpt2_untied_p4/"
    os.makedirs(OUT_DIR, exist_ok=True)

    param_grid = dict(seed=[42],
                      model_type=["gpt2"],
                      model_name_or_path=["gpt2"],
                      train_data_file=[TRAIN_FILE],
                      n_partitions=[4],
                      partitioning_batch_size=[4],
                      analysis_batch_size=[4],
                      block_size=[-1],
                      n_iter=N_ITER,
                      lmhead=[""],
                      async_pipeline=[""],
                      auto_file_name=[""],
                      overwrite_cache=[""],
                      bwd_to_fwd_ratio=TRANSFORMER_USED_RATIO,
                      output_file=[OUT_DIR])
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)
    find_best(OUT_DIR)


###################
# Bert [WIP]
###################


def bert_p4():
    raise NotImplementedError()
    COMMAND = "python partition_squad_models.py"
    TRAIN_FILE = "wikitext-2-raw/wiki.train.raw"
    OUT_DIR = "results/bert_p4/"
    os.makedirs(OUT_DIR, exist_ok=True)

    param_grid = dict(seed=[42],
                      model_type=["gpt2"],
                      model_name_or_path=["gpt2"],
                      train_data_file=[TRAIN_FILE],
                      n_partitions=[4],
                      partitioning_batch_size=[4],
                      analysis_batch_size=[4],
                      block_size=[-1],
                      n_iter=N_ITER,
                      lmhead=[""],
                      async_pipeline=[""],
                      auto_file_name=[""],
                      overwrite_cache=[""],
                      bwd_to_fwd_ratio=TRANSFORMER_USED_RATIO,
                      output_file=[OUT_DIR])
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=1)
    find_best(OUT_DIR)


if __name__ == "__main__":

    gpt2_tied_p4()
    gpt2_untied_p4()
    gpt2xl_tied_p8()
    gpt2xl_untied_p8()