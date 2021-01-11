from pipe.run.helper import RunGridHelper
from string import Template

if __name__ == '__main__':
    helper = RunGridHelper(gpu_list=list(range(8)))
    c = Template("OMP_NUM_THREADS=10 mpirun -np $nprocs python -m pipe.main --config $config")
    c_mp = Template("python -m pipe.main --config $config --nprocs $nprocs --mode mp")
    seeds = [42, 20202020, 77777777, 314159, 1322019]
    sanity_check_seed=[42]
    remaining_seeds = seeds[1:]


    cfg1 = "pipe/configs/bert/squad/bert_large_uncased_wwm_2p/stale.json"
    cfg2 = "pipe/configs/bert/squad/bert_large_uncased_wwm_2p/pipedream.json"
    cfg3 = "pipe/configs/bert/squad/bert_large_uncased_wwm_2p/gpipe.json"
    cfg4 = "pipe/configs/bert/squad/bert_large_uncased_wwm_2p/aggmsnag.json"


    cfg5 = "pipe/configs/bert/squad/bert_large_uncased_wwm_8p/stale.json"
    cfg6 = "pipe/configs/bert/squad/bert_large_uncased_wwm_8p/pipedream.json"
    cfg7 = "pipe/configs/bert/squad/bert_large_uncased_wwm_8p/gpipe.json"
    cfg8 = "pipe/configs/bert/squad/bert_large_uncased_wwm_8p/aggmsnag.json"


    cfg9 = "pipe/configs/bert/squad/bert_large_uncased_wwm_4p/stale.json"
    cfg10 = "pipe/configs/bert/squad/bert_large_uncased_wwm_4p/pipedream.json"
    cfg11 = "pipe/configs/bert/squad/bert_large_uncased_wwm_4p/gpipe.json"
    cfg12 = "pipe/configs/bert/squad/bert_large_uncased_wwm_4p/aggmsnag.json"

    cmd1 = c.substitute(dict(nprocs=2, config=cfg1))
    cmd2 = c.substitute(dict(nprocs=2, config=cfg2))
    cmd3 = c.substitute(dict(nprocs=2, config=cfg3))
    cmd4 = c.substitute(dict(nprocs=2, config=cfg4))

    # cmd5 = c.substitute(dict(nprocs=8, config=cfg5))
    # cmd6 = c.substitute(dict(nprocs=8, config=cfg6))
    # cmd7 = c.substitute(dict(nprocs=8, config=cfg7))
    # cmd8 = c.substitute(dict(nprocs=8, config=cfg8))

    # using multiprocessing comm handler
    cmd5 = c_mp.substitute(dict(nprocs=8, config=cfg5))
    cmd6 = c_mp.substitute(dict(nprocs=8, config=cfg6))
    cmd7 = c_mp.substitute(dict(nprocs=8, config=cfg7))
    cmd8 = c_mp.substitute(dict(nprocs=8, config=cfg8))


    cmd9 = c_mp.substitute(dict(nprocs=4, config=cfg9))
    cmd10 = c_mp.substitute(dict(nprocs=4, config=cfg10))
    cmd11 = c_mp.substitute(dict(nprocs=4, config=cfg11))
    cmd12 = c_mp.substitute(dict(nprocs=4, config=cfg12))

    # helper.add_runs(base_command=cmd1, param_grid={'seed': remaining_seeds}, num_gpus=2)
    # helper.add_runs(base_command=cmd2, param_grid={'seed': remaining_seeds}, num_gpus=2)
    # helper.add_runs(base_command=cmd3, param_grid={'seed': seeds}, num_gpus=2)
    # helper.add_runs(base_command=cmd4, param_grid={'seed': seeds}, num_gpus=2)


    # helper.add_runs(base_command=cmd5, param_grid={'seed': seeds}, num_gpus=8)
    # helper.add_runs(base_command=cmd6, param_grid={'seed': seeds}, num_gpus=8)
    # helper.add_runs(base_command=cmd7, param_grid={'seed': seeds}, num_gpus=8)
    # helper.add_runs(base_command=cmd8, param_grid={'seed': seeds}, num_gpus=8)


    # helper.add_runs(base_command=cmd5, param_grid={'seed': seeds}, num_gpus=8)
    # helper.add_runs(base_command=cmd6, param_grid={'seed': remaining_seeds}, num_gpus=8)
    # helper.add_runs(base_command=cmd7, param_grid={'seed': remaining_seeds}, num_gpus=8)
    # helper.add_runs(base_command=cmd8, param_grid={'seed': remaining_seeds}, num_gpus=8)


    # helper.add_runs(base_command=cmd9 + " --master_port 29500", param_grid={'seed': sanity_check_seed}, num_gpus=4)
    helper.add_runs(base_command=cmd10 + " --master_port 29501", param_grid={'seed': sanity_check_seed}, num_gpus=4)

    helper.run()
