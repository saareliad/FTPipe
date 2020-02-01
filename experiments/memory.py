import sys
sys.path.append("../")
from parsers import BatchInfoExpParser
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from exp_utils import dp_model, single_gpu, pipeline_model, get_batch_and_target, display_system_info


def run_experiment(args, model_config):
    model = model_config(args)
    optimizer = RMSprop(model.parameters())
    batch, target = get_batch_and_target(args)
    if model_config is single_gpu:
        target = target.to(batch.device)

    param_count = sum(p.storage().size() for p in model.parameters())
    param_size = sum(p.storage().size() * p.storage().element_size()
                     for p in model.parameters())
    param_scale = 3  # param + grad + RMSprop.quare_avg

    print(f'# of Model Parameters: {param_count:,}')
    print(
        f'Total Model Parameter Memory: {param_size*param_scale:,} Bytes')

    for _ in range(2):
        torch.cuda.empty_cache()
        for d in args.device_ids:
            torch.cuda.reset_max_memory_cached(d)
        if model_config is pipeline_model:
            output = model(batch, num_chunks=args.num_micro_batches)
            loss = F.cross_entropy(output, target)
            grads = torch.autograd.grad(loss, output)
            model.backward(grads)
        else:
            output = model(batch)
            loss = F.cross_entropy(output, target)
            loss.backward()
        optimizer.step()

    max_memory = 0
    for d in args.device_ids:
        torch.cuda.synchronize(d)
        max_memory += torch.cuda.max_memory_cached(d)

    return max_memory, param_size * param_count


def main(args):
    configs = []
    if not args.no_single:
        configs.append(single_gpu)
    if not args.no_dp:
        configs.append(dp_model)
    configs.append(pipeline_model)

    print('Memory experiment')

    display_system_info(args)

    for exp_config in configs:
        print()
        print(exp_config.__name__)
        try:
            max_memory, total_param_storage = run_experiment(args, exp_config)

            latent_size = max_memory - total_param_storage
            print(f'Peak Activation Memory: {latent_size:,} Bytes')
            print(f'Total Memory: {max_memory:,} Bytes')

            for d in args.device_ids:
                memory_usage = torch.cuda.memory_cached(d)
                print(f'{d!s}: {memory_usage:,} Bytes')
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('ran out of memory')
            else:
                raise e
        finally:
            print()


if __name__ == '__main__':
    parser = BatchInfoExpParser()
    args = parser.parse_args()
    main(args)
