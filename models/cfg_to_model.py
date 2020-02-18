import importlib
from .normal import WideResNet, Bottleneck, ResNet

_PARTITIONED_MODELS_PACKAGE = "models.partitioned"

_RESENETS = dict(resnet50_imagenet_p8=dict(
    block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000))

_WIDE_RESNETS = dict(
    wrn_16x4_p2=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    wrn_16x4_p4=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    wrn_28x10_c100_dr03_p4=dict(depth=28,
                                num_classes=100,
                                widen_factor=10,
                                drop_rate=0.3),
    wrn_16x4_c100_p2=dict(depth=16,
                          num_classes=100,
                          widen_factor=4,
                          drop_rate=0.0),
    wrn_16x4_c100_p4=dict(depth=16,
                          num_classes=100,
                          widen_factor=4,
                          drop_rate=0.0),
    wrn_28x10_c100_dr03_p2=dict(depth=28,
                                num_classes=100,
                                widen_factor=10,
                                drop_rate=0.3),
    # wrn_16x4_c10=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    # wrn_28x10_c10_dr03=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0.3),
    # wrn_28x10_c10=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0),
    # wrn_28x10_c100_p2
    # wrn_28x10_c100_dr03=dict(depth=28, num_classes=100, widen_factor=10, drop_rate=0.3),
    # wrn_28x10_c100=dict(depth=28, num_classes=100, widen_factor=10, drop_rate=0),
)

# MODEL_CFG_TO_SAMPLE_MODEL = {
#     **{k: WideResNet
#        for k in _WIDE_RESNETS.keys()},
#     **{k: ResNet
#        for k in _RESENETS.keys()}
# }
# MODEL_CONFIGS = {**_WIDE_RESNETS, **_RESENETS}
# CFG_TO_GENERATED_FILE_NAME = {i: i for i in MODEL_CONFIGS.keys()}

MODEL_CFG_TO_SAMPLE_MODEL = {}
MODEL_CONFIGS = {}
CFG_TO_GENERATED_FILE_NAME = {}


def _register_model(dict_params, model_cls):
    global MODEL_CFG_TO_SAMPLE_MODEL
    global MODEL_CONFIGS
    global CFG_TO_GENERATED_FILE_NAME

    MODEL_CONFIGS.update(dict_params)
    MODEL_CFG_TO_SAMPLE_MODEL.update(
        {k: model_cls
         for k in dict_params.keys()})

    CFG_TO_GENERATED_FILE_NAME = {i: i for i in MODEL_CONFIGS.keys()}


_register_model(_WIDE_RESNETS, WideResNet)
_register_model(_RESENETS, ResNet)


def create_normal_model_instance(cfg):
    """ Example : cfg='wrn_16x4' """
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg](**MODEL_CONFIGS[cfg])


def normal_model_class(cfg):
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg]


def get_partitioning(cfg, model_instance=None):
    # Import generated model
    generated_file_name = CFG_TO_GENERATED_FILE_NAME[cfg]
    generated = importlib.import_module("." + generated_file_name,
                                        package=_PARTITIONED_MODELS_PACKAGE)

    createConfig = generated.createConfig if hasattr(
        generated, "createConfig") else generated.create_pipeline_configuration

    # Create instance of normal model
    if model_instance:
        model_cls = normal_model_class(cfg)
        assert isinstance(model_instance, model_cls)
    else:
        model_instance = create_normal_model_instance(cfg)

    # Get Config
    # Explicitly ugly
    ON_CPU = True
    configs = createConfig(model_instance, partitions_only=False, DEBUG=ON_CPU)

    return configs


# if __name__ == "__main__":
#     get_partitioning('wrn_16x4')
#     pass
# Unittest
# get_partitioning('wrn_16x4')
