import torch

from autopipe.models.normal import WideResNet, amoebanetd, vgg16_bn
from autopipe.models.normal.vision_models import ResNet
from . import register_task, Parser
from .partitioning_task import PartitioningTask

# model = torch.hub.load('facebookresearch/WSL-Images', "resnext101_32x48d")
_HUB = dict(resnext101_32x48d_wsl=dict(github='facebookresearch/WSL-Images', model="resnext101_32x48d_wsl"),
            vit_large_patch16_224=dict(github='rwightman/pytorch-image-models', model='vit_large_patch16_224',
                                       pretrained=True)
            )

_VGG16_BN = dict(vgg16_bn=dict())

_RESENETS = dict(resnet50_imagenet=dict(block=ResNet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        num_classes=1000),
                 resnet101_imagenet=dict(block=ResNet.Bottleneck,
                                         layers=[3, 4, 23, 3],
                                         num_classes=1000))

_WIDE_RESNETS = dict(
    wrn_16x4=dict(depth=16, num_classes=10, widen_factor=4,
                  drop_rate=0.0),  # FOR BACKWARD COMPATABILITY
    wrn_16x4_c10=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    wrn_16x4_c100=dict(depth=16,
                       num_classes=100,
                       widen_factor=4,
                       drop_rate=0.0),
    wrn_28x10_c10_dr03=dict(depth=28,
                            num_classes=10,
                            widen_factor=10,
                            drop_rate=0.3),
    wrn_28x10_c10=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0),
    wrn_28x10_c100_dr03=dict(depth=28,
                             num_classes=100,
                             widen_factor=10,
                             drop_rate=0.3),
    wrn_28x10_c100=dict(depth=28,
                        num_classes=100,
                        widen_factor=10,
                        drop_rate=0),
)

# this model is realy big even with 4 cells it contains 845 layers
_AMOEBANET_D = dict(amoebanet_4x512_c10=dict(num_layers=4,
                                             num_filters=512,
                                             num_classes=10),
                    amoebanet_8x512_c100=dict(num_layers=8,
                                              num_filters=512,
                                              num_classes=100))

MODEL_CFG_TO_SAMPLE_MODEL = {}
MODEL_CONFIGS = {}


def _register_model(dict_params, model_cls):
    global MODEL_CFG_TO_SAMPLE_MODEL
    global MODEL_CONFIGS
    # global CFG_TO_GENERATED_FILE_NAME

    MODEL_CONFIGS.update(dict_params)
    MODEL_CFG_TO_SAMPLE_MODEL.update(
        {k: model_cls
         for k in dict_params.keys()})


_register_model(_WIDE_RESNETS, WideResNet)
_register_model(_RESENETS, ResNet.ResNet)
_register_model(_AMOEBANET_D, amoebanetd)
_register_model(_VGG16_BN, vgg16_bn)
_register_model(_HUB, torch.hub.load)

DATASETS = ['cifar10', 'cifar100', 'imagenet']


class ParsePartitioningOptsVision(Parser):
    def _add_model_args(self, group):
        group.add_argument('--model',
                           choices=MODEL_CONFIGS.keys(),
                           default='wrn_16x4')

    def _add_data_args(self, group):
        group.add_argument('-d',
                           '--dataset',
                           choices=DATASETS,
                           default='cifar10')

    def _default_values(self):
        return {
            # "model": 'wrn_16x4',
            "partitioning_batch_size": 128,
            "n_iter": 100,
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 32,
        }

    def _auto_file_name(self, args) -> str:
        return f"{args.model}_p{args.n_partitions}"


class VisionPartioner(PartitioningTask):
    def get_model(self, args) -> torch.nn.Module:
        return MODEL_CFG_TO_SAMPLE_MODEL[args.model](**MODEL_CONFIGS[args.model]).train()

    @property
    def batch_dim(self) -> int:
        return 0

    def get_input(self, args, analysis=False):
        dataset = args.dataset
        assert dataset in DATASETS
        if analysis:
            batch_size = args.analysis_batch_size
        else:
            batch_size = args.partitioning_batch_size

        # TODO: just choose size like we do for seq len
        if dataset == 'cifar10' or dataset == 'cifar100':
            sample = torch.randn(batch_size, 3, 32, 32)
        elif dataset == 'imagenet':
            sample = torch.randn(batch_size, 3, 224, 224)
        else:
            raise NotImplementedError()
        return sample


register_task("vision", ParsePartitioningOptsVision, VisionPartioner)
