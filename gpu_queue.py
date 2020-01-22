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


def map_to_limited_gpus(func, configs, N_GPU, CUDA_VISIBLE_DEVICES=None):
    with Manager() as manager:
        q = manager.Queue()

        if CUDA_VISIBLE_DEVICES:
            for i in CUDA_VISIBLE_DEVICES:
                q.put(i)
        else:
            # TODO: if os.environ.get('CUDA_VISIBLE_DEVICES'):
            for i in range(N_GPU):
                q.put(i)

        Parallel(n_jobs=N_GPU, verbose=10)(
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
