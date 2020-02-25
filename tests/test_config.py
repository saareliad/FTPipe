from models.normal import dummy
from pytorch_Gpipe.utils import layerDict, tensorDict
from pytorch_Gpipe import PipelineConfig
import torch


def test_config():
    base_model = dummy.Dummy()
    tensors = tensorDict(base_model)
    layers = layerDict(base_model)

    config = PipelineConfig().add_input(
        "input0").add_output("output0").add_output("output1").add_output("output2")

    stage0 = config.add_stage(dummy.Stage0, torch.optim.SGD, {
        'lr': 1e-3}).add_input("input0").add_output("output2").add_devices('cpu')

    config.add_stage(dummy.Stage1).add_input("input0").add_output(
        "t0").add_devices('cpu', 'cpu')

    stage2 = config.add_stage(
        dummy.Stage2).add_input("t0").add_output("t1")
    for _ in range(4):
        stage2.add_devices('cpu')

    stage3 = config.add_stage(dummy.Stage3).add_input(
        "t1").add_output("output0").add_output("output1").add_devices('cpu', 'cuda:0')
    assert config.isValid()
    L, R = config.split({1, 3})
    assert config.isValid()
    assert L.isValid()
    assert R.isValid()
    assert PipelineConfig.fromJson(config.toJson()).isValid()
