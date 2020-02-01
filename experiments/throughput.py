import sys
sys.path.append("../")
from parsers import BatchInfoExpParser
import torch
import torch.nn.functional as F
from torch.optim import SGD
from exp_utils import dp_model, single_gpu, pipeline_model, get_batch_and_target, display_system_info
import time


def run_experiment(args, model_config):
    model = model_config(args)
    optimizer = SGD(model.parameters(), lr=0.1)
    batch, target = get_batch_and_target(args)
    if model_config is single_gpu:
        target = target.to(batch.device)

    dataset_size = 50000
    data = [(batch, target)] * (dataset_size // args.batch_size)
    throughputs = []
    epoch_times = []

    for epoch in range(args.epochs):
        data_trained = 0
        torch.cuda.synchronize(args.device_ids[0])
        start = time.time()

        for input, target in data:
            if model_config is pipeline_model:
                output = model(input, num_chunks=args.num_micro_batches)
                loss = F.cross_entropy(output, target)
                grad = torch.autograd.grad(loss, output)
                model.backward(grad)
            else:
                output = model(input)
                loss = F.cross_entropy(output, target)
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            data_trained += target.size(0)

        torch.cuda.synchronize(args.device_ids[0])
        epoch_time = time.time() - start
        throughput = data_trained / epoch_time

        print(f"epoch[{epoch+1}/{args.epochs+1}] throughput: {throughput:.2f}samples/second epoch time {epoch_time:.3f}seconds/epoch")

        if epoch >= args.warmup_epochs:
            throughputs.append(throughput)
            epoch_times.append(epoch_time)

    return throughputs, epoch_times


def main(args):
    configs = []
    if not args.no_single:
        configs.append(single_gpu)
    if not args.no_dp:
        configs.append(dp_model)
    configs.append(pipeline_model)

    print('Throughput experiment')

    display_system_info(args)
    for exp_config in configs:
        print()
        print(exp_config.__name__)
        try:
            throughputs, epoch_times = run_experiment(args, exp_config)

            avg_throughput = sum(throughputs) / len(throughputs)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)

            print(
                f"average throughput: {avg_throughput:.2f}samples/second average epoch time {avg_epoch_time:.3f}seconds/epoch")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('ran out of memory')
            else:
                raise e
        finally:
            print()


if __name__ == "__main__":
    parser = BatchInfoExpParser()
    parser.add_argument(
        "--epochs", type=int, help="number of epochs to run the experiment", default=10)
    parser.add_argument("--warmup_epochs", type=int,
                        help="number of warmup epochs to run before measuring", default=1)
    args = parser.parse_args()
    main(args)
