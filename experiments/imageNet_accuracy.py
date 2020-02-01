import sys
sys.path.append("../")
import time
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision

from exp_utils import dp_model, single_gpu, pipeline_model, display_system_info
from parsers import ImageNetParser


from sample_models import alexnet, resnet101, vgg19_bn, squeezenet1_1, densenet201, \
    WideResNet, AmoebaNet_D

MODELS = {
    "alexnet": alexnet,
    "resnet101": resnet101,
    "vgg19_bn": vgg19_bn,
    "squeezenet1_1": squeezenet1_1,
    "densenet201": densenet201,
    "WideResNet": WideResNet,
    "amoebanet": AmoebaNet_D,
}


def dataloaders(batch_size: int, num_workers: int = 32):
    num_workers = num_workers if batch_size <= 4096 else num_workers // 2

    post_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = torchvision.datasets.ImageNet(
        root='imagenet',
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            post_transforms,
        ])
    )
    test_dataset = torchvision.datasets.ImageNet(
        root='imagenet',
        split='val',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            post_transforms,
        ])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    nesterov = args.nesterov
    momentum = args.momentum

    if args.dp:
        model = dp_model(args)
        experiment = "single gpu"
    elif args.single:
        model = single_gpu(args)
        experiment = "data parallel"
    else:
        model = pipeline_model(args)
        experiment = "pipeline"

    # Prepare dataloaders.
    train_dataloader, valid_dataloader = dataloaders(
        batch_size, args.num_workers)

    # Optimizer with LR scheduler
    steps = len(train_dataloader)
    lr_multiplier = max(1.0, batch_size / 256)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,
                    weight_decay=weight_decay, nesterov=nesterov)

    def gradual_warmup_linear_scaling(step: int):
        epoch = step / steps

        # Gradual warmup
        warmup_ratio = min(4.0, epoch) / 4.0
        multiplier = warmup_ratio * (lr_multiplier - 1.0) + 1.0

        if epoch < 30:
            return 1.0 * multiplier
        elif epoch < 60:
            return 0.1 * multiplier
        elif epoch < 80:
            return 0.01 * multiplier
        return 0.001 * multiplier

    scheduler = LambdaLR(optimizer, lr_lambda=gradual_warmup_linear_scaling)

    in_device = args.device_ids[0]
    out_device = args.device_ids[-1]

    # HEADER ======================================================================================

    title = '%s, %d devices, %d batch, %d epochs'\
            '' % (experiment, len(args.device_ids),
                  batch_size, epochs)
    print(title)
    display_system_info(args)

    # TRAIN =======================================================================================

    def evaluate(dataloader):
        tick = time.time()
        steps = len(dataloader)
        data_tested = 0
        loss_sum = torch.zeros(1, device=out_device)
        accuracy_sum = torch.zeros(1, device=out_device)
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(dataloader):
                current_batch = input.size(0)
                data_tested += current_batch
                input = input.to(device=in_device)
                target = target.to(device=out_device)

                if args.pipeline:
                    output = model(input, args.num_micro_batches)
                else:
                    output = model(input)
                loss = F.cross_entropy(output, target)
                loss_sum += loss * current_batch

                _, predicted = torch.max(output, 1)
                correct = (predicted == target).sum()
                accuracy_sum += correct

                percent = i / steps * 100
                throughput = data_tested / (time.time() - tick)
                print('valid | %d%% | %.3f samples/sec (estimated)'
                      '' % (percent, throughput), clear=True, nl=False)

        loss = loss_sum / data_tested
        accuracy = accuracy_sum / data_tested

        return loss.item(), accuracy.item()

    def run_epoch(epoch: int):
        torch.cuda.synchronize(in_device)
        tick = time.time()

        steps = len(train_dataloader)
        data_trained = 0
        loss_sum = torch.zeros(1, device=out_device)
        model.train()
        for i, (input, target) in enumerate(train_dataloader):
            data_trained += batch_size
            input = input.to(device=in_device, non_blocking=True)
            target = target.to(device=out_device, non_blocking=True)
            optimizer.zero_grad()
            if args.pipeline:
                output = model(input, args.num_micro_batches)
                loss = F.cross_entropy(output, target)
                model.backward(torch.autograd.grad(loss, output))
            else:
                output = model(input)
                loss = F.cross_entropy(output, target)
                loss.backward()

            optimizer.step()
            scheduler.step()

            loss_sum += loss.detach() * batch_size

            percent = i / steps * 100
            throughput = data_trained / (time.time() - tick)
            print('train | %d/%d epoch (%d%%) | lr:%.5f | %.3f samples/sec (estimated)'
                  '' % (epoch + 1, epochs, percent,
                        scheduler.get_lr()[0], throughput),
                  clear=True, nl=False)

        torch.cuda.synchronize(in_device)
        tock = time.time()

        train_loss = loss_sum.item() / data_trained
        valid_loss, valid_accuracy = evaluate(valid_dataloader)
        torch.cuda.synchronize(in_device)

        elapsed_time = tock - tick
        throughput = data_trained / elapsed_time
        print('%d/%d epoch | lr:%.5f | train loss:%.3f %.3f samples/sec | '
              'valid loss:%.3f accuracy:%.3f'
              '' % (epoch + 1, epochs, scheduler.get_lr()[0], train_loss, throughput,
                    valid_loss, valid_accuracy),
              clear=True)

        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []

    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(epoch)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    _, valid_accuracy = evaluate(valid_dataloader)

    # RESULT ======================================================================================

    # pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('%s | valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
          '' % (title, valid_accuracy, throughput, elapsed_time))


if __name__ == '__main__':
    parser = ImageNetParser(MODELS)
    args = parser.parse_args()
    main(args)
