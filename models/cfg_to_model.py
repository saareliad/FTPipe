import importlib
from .normal import WideResNet, WideResNetNonInplace

_PARTITIONED_MODELS_PACKAGE = "models.partitioned"
_WIDE_RESNETS = dict(
    wrn_16x4_p2=dict(depth=16, num_classes=10, widen_factor=4,
                     drop_rate=0.0),
    wrn_16x4_p4=dict(depth=16, num_classes=10, widen_factor=4,
                     drop_rate=0.0),
    wrn_28x10_c100_dr03_p4=dict(
        depth=28, num_classes=100, widen_factor=10, drop_rate=0.3),
    # wrn_16x4_c10=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    # wrn_28x10_c10_dr03=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0.3),
    # wrn_28x10_c10=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0),

    # wrn_16x4_c100=dict(depth=16, num_classes=100, widen_factor=4, drop_rate=0.0),
    # wrn_28x10_c100_dr03=dict(depth=28, num_classes=100, widen_factor=10, drop_rate=0.3),
    # wrn_28x10_c100=dict(depth=28, num_classes=100, widen_factor=10, drop_rate=0),
)
_NON_INPLACE_WIDE_RESNETS = dict(
    # wrn_28x10_c100_dr03_p4=dict(
    #     depth=28, num_classes=100, widen_factor=10, drop_rate=0.3),
)
MODEL_CFG_TO_SAMPLE_MODEL = {**{k: WideResNet for k in _WIDE_RESNETS.keys()},
                             **{k: WideResNetNonInplace for k in _NON_INPLACE_WIDE_RESNETS.keys()},
                             }
MODEL_CONFIGS = {**_WIDE_RESNETS, **_NON_INPLACE_WIDE_RESNETS}
CFG_TO_GENERATED_FILE_NAME = {i: i for i in MODEL_CONFIGS.keys()}


def create_normal_model_instance(cfg):
    """ Example : cfg='wrn_16x4' """
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg](**MODEL_CONFIGS[cfg])


def normal_model_class(cfg):
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg]


def get_partitioning(cfg, model_instance=None):
    # Import generated model
    generated_file_name = CFG_TO_GENERATED_FILE_NAME[cfg]
    generated = importlib.import_module(
        "." + generated_file_name, package=_PARTITIONED_MODELS_PACKAGE)
    createConfig = generated.createConfig

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

# Unittest
# get_partitioning('wrn_16x4')
