from joblib import Parallel, delayed
from multiprocessing import Manager
import os


def run_function(func, cfg, q):
    gpu = q.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(f"GPU:{gpu}")
    func(**cfg)

    q.put(gpu)


def run_function_on_several_gpus(required_gpus, func, cfg, q):
    gpus = [q.get() for _ in range(required_gpus)]
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpus)
    print(f"GPUs:{gpus}")
    func(**cfg)

    for gpu in gpus:
        q.put(gpu)

# TODO: this is not working yet...


def map_to_several_limited_gpus(func, configs, gpus_per_config, NUM_AVAIALBLE_GPUS, CUDA_VISIBLE_DEVICES=None):
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

        if not isinstance(gpus_per_config, list):
            gpus_per_config = [gpus_per_config for _ in range(len(configs))]

        assert len(gpus_per_config) == len(configs)

        n_jobs = NUM_AVAIALBLE_GPUS // max(gpus_per_config)

        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_function_on_several_gpus)(required_gpus, func, cfg, q)
            for cfg, required_gpus in zip(configs, gpus_per_config))


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
