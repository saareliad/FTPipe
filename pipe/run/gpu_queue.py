import os
from multiprocessing import Manager

from joblib import Parallel, delayed


def run_function(func, cfg, q):
    gpu = q.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(f"# GPU:{gpu}")
    func(**cfg)

    q.put(gpu)


def run_function_on_several_gpus(required_gpus, func, cfg, q):
    gpus = [str(q.get()) for _ in range(required_gpus)]
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpus)
    print(f"# GPUs:{gpus}")
    func(**cfg)

    for gpu in gpus:
        q.put(gpu)


def prepare_gpu_queue(manager, NUM_AVAIALBLE_GPUS, CUDA_VISIBLE_DEVICES=None):
    q = manager.Queue()
    # Mark GPUs as avaialbale
    if CUDA_VISIBLE_DEVICES:
        for i in CUDA_VISIBLE_DEVICES:
            q.put(i)
    else:
        # TODO: if os.environ.get('CUDA_VISIBLE_DEVICES'):
        for i in range(NUM_AVAIALBLE_GPUS):
            q.put(i)
    return q


def map_to_several_limited_gpus(func, configs, gpus_per_config, NUM_AVAIALBLE_GPUS, CUDA_VISIBLE_DEVICES=None):
    with Manager() as manager:
        q = prepare_gpu_queue(manager, NUM_AVAIALBLE_GPUS, CUDA_VISIBLE_DEVICES)

        if not isinstance(gpus_per_config, list):
            gpus_per_config = [gpus_per_config for _ in range(len(configs))]

        assert len(gpus_per_config) == len(configs)

        n_jobs = NUM_AVAIALBLE_GPUS // max(gpus_per_config)

        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_function_on_several_gpus)(required_gpus, func, cfg, q)
            for cfg, required_gpus in zip(configs, gpus_per_config))


def pop_from_cfg(cfg, name):
    attr = cfg.pop(name)
    return cfg, attr


def pop_FUNC_from_cfg(cfg):
    return pop_from_cfg(cfg, name='FUNC')


def pop_REQUIRED_GPUS_from_cfg(cfg):
    return pop_from_cfg(cfg, name='REQUIRED_GPUS')


def flexible_map_to_several_limited_gpus(configs, NUM_AVAIALBLE_GPUS, CUDA_VISIBLE_DEVICES=None):
    with Manager() as manager:
        q = prepare_gpu_queue(manager, NUM_AVAIALBLE_GPUS, CUDA_VISIBLE_DEVICES)
        gpus_per_config = []
        funcs = []
        cfgs = []
        for cfg in configs:
            cfg, f = pop_FUNC_from_cfg(cfg)
            cfg, r = pop_REQUIRED_GPUS_from_cfg(cfg)
            funcs.append(f)
            gpus_per_config.append(r)
            cfgs.append(cfg)
        configs = cfgs
        n_jobs = NUM_AVAIALBLE_GPUS // max(gpus_per_config)
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_function_on_several_gpus)(required_gpus, func, cfg, q)
            for cfg, required_gpus, func in zip(configs, gpus_per_config, funcs))


def map_to_limited_gpus(func, configs, NUM_AVAIALBLE_GPUS, CUDA_VISIBLE_DEVICES=None):
    with Manager() as manager:
        q = manager.Queue()

        # Mark GPUs as avaialbale
        if CUDA_VISIBLE_DEVICES:
            for i in CUDA_VISIBLE_DEVICES:
                q.put(i)
        else:
            # TODO: if os.environ.get('CUDA_VISIBLE_DEVICES'):
            for i in range(NUM_AVAIALBLE_GPUS):
                q.put(i)

        Parallel(n_jobs=NUM_AVAIALBLE_GPUS, verbose=10)(
            delayed(run_function)(func, cfg, q) for cfg in configs)


if __name__ == "__main__":

    def test_map_to_limited_gpus():
        def foo(**kw):
            print(kw)

        def get_configs():
            configs = []
            for i in range(100):
                configs.append({'c': 1, 'd': 2})
            return configs

        map_to_limited_gpus(foo, get_configs(), 4)


    test_map_to_limited_gpus()
