import torch
from torch.nn import DataParallel
import sys
sys.path.append("../")
from pytorch_Gpipe import Pipeline
import platform


def dp_model(args):
    model = args.model_class(**args.model_args)
    gpu_ids = args.device_ids
    return DataParallel(model.to(gpu_ids[0]), device_ids=gpu_ids, output_device=gpu_ids[0])


def pipeline_model(args):
    model = args.model_class(**args.model_args)
    config = args.pipeline_config(model, DEBUG=True, partitions_only=False)
    gpu_ids = args.device_ids

    for idx, gpu in enumerate(gpu_ids):
        config[idx].device = gpu
        config[idx] = config[idx].to(gpu)

    return Pipeline(config, gpu_ids[-1])


def single_gpu(args):
    return args.model_class(**args.model_args).to(args.device_ids[0])


def get_batch_and_target(args, dp=False):
    for k in ['gpu_batch_size', 'batch_shape', 'batch_dim']:
        assert k in args
    if dp:
        batch_size = args.gpu_batch_size // len(args.device_ids)
    else:
        batch_size = args.gpu_batch_size

    dim = args.batch_dim
    shape = args.batch_shape

    shape = shape[:dim] + [batch_size] + shape[dim:]

    return torch.randn(shape, device=args.device_ids[0]), torch.randint(100, (batch_size,), device=args.device_ids[-1])


def display_system_info(args):
    print('python: %s, torch: %s, cudnn: %s, cuda: %s, gpus: %s' % (
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda,
        ",".join(torch.cuda.get_device_name(idx) for idx in args.device_ids)))
