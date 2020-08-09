from .normal.WideResNet_GN import WideResNet as WideResNet_GN
from .normal import WideResNet, Bottleneck, ResNet

from .models import register_model


register_model(name='wrn_28x10_c100_dr03_p4',
               dict_params=dict(depth=28,
                                num_classes=100,
                                widen_factor=10,
                                drop_rate=0.3),
               model_class=WideResNet,
               generated_file_name_or_path='wrn_28x10_c100_dr03_p4')


register_model(name='wrn_28x10_c100_dr03_p4_group_norm',
               dict_params=dict(depth=28,
                                num_classes=100,
                                widen_factor=10,
                                drop_rate=0.3),
               model_class=WideResNet_GN,
               generated_file_name_or_path='wrn_28x10_c100_dr03_p4_group_norm')


register_model(name='resnet50_imagenet_p8',
               dict_params=dict(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000),
               model_class=ResNet,
               generated_file_name_or_path='resnet50_imagenet_p8')
