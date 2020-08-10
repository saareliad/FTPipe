from .normal.WideResNet_GN import WideResNet as WideResNet_GN
from .normal import WideResNet, Bottleneck, ResNet

from .models import register_model
from .models import CommonModelHandler


class ParamDictCVMOdelHandler(CommonModelHandler):
    def __init__(self, dict_params, model_class, *args, **kw):
        super().__init__(*args, **kw)
        self.dict_params = dict_params
        self.model_class = model_class

    def get_normal_model_instance(self, *args, **kw):
        return self.model_class(self.dict_params)

    def get_loader(self, *args, **kw):
        raise NotImplementedError()


def register_cv_hardcoded_model(name, *args, **kw):
    handler = ParamDictCVMOdelHandler(*args, **kw)
    register_model(name, handler=handler)


register_cv_hardcoded_model(name='wrn_28x10_c100_dr03_p4',
                            dict_params=dict(depth=28,
                                             num_classes=100,
                                             widen_factor=10,
                                             drop_rate=0.3),
                            model_class=WideResNet,
                            generated_file_name_or_path='wrn_28x10_c100_dr03_p4')


register_cv_hardcoded_model(name='wrn_28x10_c100_dr03_p4_group_norm',
               dict_params=dict(depth=28,
                                num_classes=100,
                                widen_factor=10,
                                drop_rate=0.3),
               model_class=WideResNet_GN,
               generated_file_name_or_path='wrn_28x10_c100_dr03_p4_group_norm')


register_cv_hardcoded_model(name='resnet50_imagenet_p8',
               dict_params=dict(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000),
               model_class=ResNet,
               generated_file_name_or_path='resnet50_imagenet_p8')
