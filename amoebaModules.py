import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_Gpipe.utils import layerDict, tensorDict, OrderedSet
from module_generation.pipeline import Pipeline
from torch.nn.modules.pooling import AvgPool2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Identity
from sample_models.amoebaNet import MergeTwo
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
# this is an auto generated file do not edit unless you know what you are doing

#TODO had to explicitly convert every MergeTwo call into torch.add_call
# as we actually record only the 2 tensors which are merged by the mergeTwo layer

# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {'output0'}}
# model outputs {2}

def AmoebaNet_DPipeline(model:nn.Module,output_device=None,DEBUG=False):
    layer_dict = layerDict(model)
    tensor_dict = tensorDict(model)
    
    # now constructing the partitions in order
    layer_scopes = ['AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu]',
        'AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = AmoebaNet_DPartition0(layers,buffers,parameters)

    layer_scopes = ['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = AmoebaNet_DPartition1(layers,buffers,parameters)

    layer_scopes = ['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]',
        'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14]',
        'AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling]',
        'AmoebaNet_D/Classifier[classifier]/Linear[classifier]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition2 = AmoebaNet_DPartition2(layers,buffers,parameters)

    # creating configuration
    config = {0: {'inputs': OrderedSet(['input0']), 'outputs': OrderedSet(['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/aten::cat1439', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]'])},
            1: {'inputs': OrderedSet(['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/aten::cat1439', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]']), 'outputs': OrderedSet(['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/aten::cat3746', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]'])},
            2: {'inputs': OrderedSet(['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/aten::cat3746', 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]']), 'outputs': OrderedSet(['AmoebaNet_D/Classifier[classifier]/Linear[classifier]'])}
            }
    config[0]['model'] = partition0.to('cpu')
    config[1]['model'] = partition1.to('cpu')
    config[2]['model'] = partition2.to('cpu')
    config['model inputs'] = ['input0']
    config['model outputs'] = ['AmoebaNet_D/Classifier[classifier]/Linear[classifier]']
    
    return Pipeline(config,output_device=output_device,DEBUG=DEBUG)


class AmoebaNet_DPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(AmoebaNet_DPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 104)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu]
        assert 'AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu] was expected but not given'
        self.l_0 = layers['AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu]']
        assert isinstance(self.l_0,ReLU) ,f'layers[AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_0)}'
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_1 = layers['AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_1,Conv2d) ,f'layers[AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_1)}'
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_2 = layers['AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_2,BatchNorm2d) ,f'layers[AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_2)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu] was expected but not given'
        self.l_3 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu]']
        assert isinstance(self.l_3,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_3)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv] was expected but not given'
        self.l_4 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv]']
        assert isinstance(self.l_4,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm] was expected but not given'
        self.l_5 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm]']
        assert isinstance(self.l_5,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu] was expected but not given'
        self.l_6 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu]']
        assert isinstance(self.l_6,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_6)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv] was expected but not given'
        self.l_7 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv]']
        assert isinstance(self.l_7,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_7)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm] was expected but not given'
        self.l_8 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm]']
        assert isinstance(self.l_8,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_8)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_9 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_9,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_9)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_10 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_10,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_11 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_11,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_12 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_12,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_12)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_13 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_13,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_13)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_14 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_14,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_14)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2] was expected but not given'
        self.l_15 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]']
        assert isinstance(self.l_15,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]] is expected to be of type MergeTwo but was of type {type(self.l_15)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_16 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_16,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_16)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_17 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_17,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_17)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_18 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_18,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_18)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] was expected but not given'
        self.l_19 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]']
        assert isinstance(self.l_19,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_19)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] was expected but not given'
        self.l_20 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]']
        assert isinstance(self.l_20,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_20)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] was expected but not given'
        self.l_21 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]']
        assert isinstance(self.l_21,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_21)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] was expected but not given'
        self.l_22 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]']
        assert isinstance(self.l_22,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_22)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] was expected but not given'
        self.l_23 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]']
        assert isinstance(self.l_23,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_23)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_24 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_24,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_24)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5] was expected but not given'
        self.l_25 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5]']
        assert isinstance(self.l_25,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5]] is expected to be of type MergeTwo but was of type {type(self.l_25)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_26 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_26,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_26)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_27 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_27,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_27)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_28 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_28,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_28)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_29 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_29,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_29)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_30 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_30,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_30)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_31 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_31,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_31)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_32 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_32,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_32)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_33 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_33,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_33)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_34 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_34,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_34)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_35 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_35,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_35)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_36 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_36,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_36)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_37 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_37,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_37)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_38 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_38,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_38)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_39 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_39,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_39)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_40 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_40,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_40)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8] was expected but not given'
        self.l_41 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8]']
        assert isinstance(self.l_41,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8]] is expected to be of type MergeTwo but was of type {type(self.l_41)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_42 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_42,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_42)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_43 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_43,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_43)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_44 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_44,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_44)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_45 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_45,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_45)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_46 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_46,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_46)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_47 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_47,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_47)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11] was expected but not given'
        self.l_48 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11]']
        assert isinstance(self.l_48,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11]] is expected to be of type MergeTwo but was of type {type(self.l_48)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_49 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_49,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_49)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_50 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_50,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_50)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_51 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_51,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_51)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14] was expected but not given'
        self.l_52 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14]']
        assert isinstance(self.l_52,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14]] is expected to be of type MergeTwo but was of type {type(self.l_52)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu] was expected but not given'
        self.l_53 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu]']
        assert isinstance(self.l_53,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_53)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1] was expected but not given'
        self.l_54 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1]']
        assert isinstance(self.l_54,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1]] is expected to be of type Conv2d but was of type {type(self.l_54)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2] was expected but not given'
        self.l_55 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2]']
        assert isinstance(self.l_55,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2]] is expected to be of type Conv2d but was of type {type(self.l_55)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] was expected but not given'
        self.l_56 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]']
        assert isinstance(self.l_56,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]] is expected to be of type BatchNorm2d but was of type {type(self.l_56)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu] was expected but not given'
        self.l_57 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu]']
        assert isinstance(self.l_57,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_57)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv] was expected but not given'
        self.l_58 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv]']
        assert isinstance(self.l_58,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_58)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm] was expected but not given'
        self.l_59 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm]']
        assert isinstance(self.l_59,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_59)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_60 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_60,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_60)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_61 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_61,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_61)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_62 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_62,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_62)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_63 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_63,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_63)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_64 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_64,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_64)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_65 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_65,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_65)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2] was expected but not given'
        self.l_66 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]']
        assert isinstance(self.l_66,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]] is expected to be of type MergeTwo but was of type {type(self.l_66)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_67 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_67,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_67)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_68 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_68,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_68)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_69 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_69,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_69)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] was expected but not given'
        self.l_70 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]']
        assert isinstance(self.l_70,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_70)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] was expected but not given'
        self.l_71 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]']
        assert isinstance(self.l_71,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_71)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] was expected but not given'
        self.l_72 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]']
        assert isinstance(self.l_72,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_72)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] was expected but not given'
        self.l_73 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]']
        assert isinstance(self.l_73,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_73)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] was expected but not given'
        self.l_74 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]']
        assert isinstance(self.l_74,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_74)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_75 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_75,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_75)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5] was expected but not given'
        self.l_76 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5]']
        assert isinstance(self.l_76,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5]] is expected to be of type MergeTwo but was of type {type(self.l_76)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_77 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_77,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_77)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_78 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_78,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_78)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_79 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_79,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_79)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_80 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_80,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_80)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_81 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_81,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_81)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_82 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_82,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_82)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_83 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_83,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_83)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_84 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_84,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_84)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_85 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_85,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_85)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_86 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_86,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_86)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_87 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_87,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_87)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_88 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_88,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_88)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_89 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_89,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_89)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_90 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_90,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_90)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_91 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_91,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_91)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8] was expected but not given'
        self.l_92 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]']
        assert isinstance(self.l_92,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]] is expected to be of type MergeTwo but was of type {type(self.l_92)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_93 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_93,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_93)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_94 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_94,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_94)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_95 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_95,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_95)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_96 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_96,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_96)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_97 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_97,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_97)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_98 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_98,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_98)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11] was expected but not given'
        self.l_99 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]']
        assert isinstance(self.l_99,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]] is expected to be of type MergeTwo but was of type {type(self.l_99)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_100 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_100,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_100)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_101 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_101,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_101)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_102 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_102,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_102)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14] was expected but not given'
        self.l_103 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]']
        assert isinstance(self.l_103,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]] is expected to be of type MergeTwo but was of type {type(self.l_103)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.lookup = {'l_0': 'stem.conv_cell.relu', 'l_1': 'stem.conv_cell.conv', 'l_2': 'stem.conv_cell.norm', 'l_3': 'cells.0.preprocess0.relu', 'l_4': 'cells.0.preprocess0.conv', 'l_5': 'cells.0.preprocess0.norm', 'l_6': 'cells.0.preprocess1.relu', 'l_7': 'cells.0.preprocess1.conv', 'l_8': 'cells.0.preprocess1.norm', 'l_9': 'cells.0.layers.0.module.pool', 'l_10': 'cells.0.layers.0.module.conv_cell.conv', 'l_11': 'cells.0.layers.0.module.conv_cell.norm', 'l_12': 'cells.0.layers.1.module.pool', 'l_13': 'cells.0.layers.1.module.conv_cell.conv', 'l_14': 'cells.0.layers.1.module.conv_cell.norm', 'l_15': 'cells.0.layers.2', 'l_16': 'cells.0.layers.3.module.conv1_1x1.relu', 'l_17': 'cells.0.layers.3.module.conv1_1x1.conv', 'l_18': 'cells.0.layers.3.module.conv1_1x1.norm', 'l_19': 'cells.0.layers.3.module.conv2_3x3.relu', 'l_20': 'cells.0.layers.3.module.conv2_3x3.conv', 'l_21': 'cells.0.layers.3.module.conv2_3x3.norm', 'l_22': 'cells.0.layers.3.module.conv3_1x1.relu', 'l_23': 'cells.0.layers.3.module.conv3_1x1.conv', 'l_24': 'cells.0.layers.3.module.conv3_1x1.norm', 'l_25': 'cells.0.layers.5', 'l_26': 'cells.0.layers.6.module.conv1_1x1.relu', 'l_27': 'cells.0.layers.6.module.conv1_1x1.conv', 'l_28': 'cells.0.layers.6.module.conv1_1x1.norm', 'l_29': 'cells.0.layers.6.module.conv2_1x7.relu', 'l_30': 'cells.0.layers.6.module.conv2_1x7.conv', 'l_31': 'cells.0.layers.6.module.conv2_1x7.norm', 'l_32': 'cells.0.layers.6.module.conv3_7x1.relu', 'l_33': 'cells.0.layers.6.module.conv3_7x1.conv', 'l_34': 'cells.0.layers.6.module.conv3_7x1.norm', 'l_35': 'cells.0.layers.6.module.conv4_1x1.relu', 'l_36': 'cells.0.layers.6.module.conv4_1x1.conv', 'l_37': 'cells.0.layers.6.module.conv4_1x1.norm', 'l_38': 'cells.0.layers.7.module.pool', 'l_39': 'cells.0.layers.7.module.conv_cell.conv', 'l_40': 'cells.0.layers.7.module.conv_cell.norm', 'l_41': 'cells.0.layers.8', 'l_42': 'cells.0.layers.9.module.pool', 'l_43': 'cells.0.layers.9.module.conv_cell.conv', 'l_44': 'cells.0.layers.9.module.conv_cell.norm', 'l_45': 'cells.0.layers.10.module.relu', 'l_46': 'cells.0.layers.10.module.conv', 'l_47': 'cells.0.layers.10.module.norm', 'l_48': 'cells.0.layers.11', 'l_49': 'cells.0.layers.13.module.pool', 'l_50': 'cells.0.layers.13.module.conv_cell.conv', 'l_51': 'cells.0.layers.13.module.conv_cell.norm', 'l_52': 'cells.0.layers.14', 'l_53': 'cells.1.preprocess0.relu', 'l_54': 'cells.1.preprocess0.conv_1', 'l_55': 'cells.1.preprocess0.conv_2', 'l_56': 'cells.1.preprocess0.bn', 'l_57': 'cells.1.preprocess1.relu', 'l_58': 'cells.1.preprocess1.conv', 'l_59': 'cells.1.preprocess1.norm', 'l_60': 'cells.1.layers.0.module.pool', 'l_61': 'cells.1.layers.0.module.conv_cell.conv', 'l_62': 'cells.1.layers.0.module.conv_cell.norm', 'l_63': 'cells.1.layers.1.module.pool', 'l_64': 'cells.1.layers.1.module.conv_cell.conv', 'l_65': 'cells.1.layers.1.module.conv_cell.norm', 'l_66': 'cells.1.layers.2', 'l_67': 'cells.1.layers.3.module.conv1_1x1.relu', 'l_68': 'cells.1.layers.3.module.conv1_1x1.conv', 'l_69': 'cells.1.layers.3.module.conv1_1x1.norm', 'l_70': 'cells.1.layers.3.module.conv2_3x3.relu', 'l_71': 'cells.1.layers.3.module.conv2_3x3.conv', 'l_72': 'cells.1.layers.3.module.conv2_3x3.norm', 'l_73': 'cells.1.layers.3.module.conv3_1x1.relu', 'l_74': 'cells.1.layers.3.module.conv3_1x1.conv', 'l_75': 'cells.1.layers.3.module.conv3_1x1.norm', 'l_76': 'cells.1.layers.5', 'l_77': 'cells.1.layers.6.module.conv1_1x1.relu', 'l_78': 'cells.1.layers.6.module.conv1_1x1.conv', 'l_79': 'cells.1.layers.6.module.conv1_1x1.norm', 'l_80': 'cells.1.layers.6.module.conv2_1x7.relu', 'l_81': 'cells.1.layers.6.module.conv2_1x7.conv', 'l_82': 'cells.1.layers.6.module.conv2_1x7.norm', 'l_83': 'cells.1.layers.6.module.conv3_7x1.relu', 'l_84': 'cells.1.layers.6.module.conv3_7x1.conv', 'l_85': 'cells.1.layers.6.module.conv3_7x1.norm', 'l_86': 'cells.1.layers.6.module.conv4_1x1.relu', 'l_87': 'cells.1.layers.6.module.conv4_1x1.conv', 'l_88': 'cells.1.layers.6.module.conv4_1x1.norm', 'l_89': 'cells.1.layers.7.module.pool', 'l_90': 'cells.1.layers.7.module.conv_cell.conv', 'l_91': 'cells.1.layers.7.module.conv_cell.norm', 'l_92': 'cells.1.layers.8', 'l_93': 'cells.1.layers.9.module.pool', 'l_94': 'cells.1.layers.9.module.conv_cell.conv', 'l_95': 'cells.1.layers.9.module.conv_cell.norm', 'l_96': 'cells.1.layers.10.module.relu', 'l_97': 'cells.1.layers.10.module.conv', 'l_98': 'cells.1.layers.10.module.norm', 'l_99': 'cells.1.layers.11', 'l_100': 'cells.1.layers.13.module.pool', 'l_101': 'cells.1.layers.13.module.conv_cell.conv', 'l_102': 'cells.1.layers.13.module.conv_cell.norm', 'l_103': 'cells.1.layers.14'}

    def forward(self, x0):
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu] <=> self.l_0
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_1
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_2
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu] <=> self.l_3
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv] <=> self.l_4
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm] <=> self.l_5
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu] <=> self.l_6
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv] <=> self.l_7
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm] <=> self.l_8
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_9
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_10
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_11
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_12
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_13
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_14
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2] <=> self.l_15
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_16
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_17
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_18
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] <=> self.l_19
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] <=> self.l_20
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] <=> self.l_21
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] <=> self.l_22
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] <=> self.l_23
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] <=> self.l_24
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5] <=> self.l_25
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_26
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_27
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_28
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_29
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_30
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_31
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_32
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_33
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_34
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_35
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_36
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_37
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_38
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_39
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_40
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8] <=> self.l_41
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_42
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_43
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_44
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] <=> self.l_45
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] <=> self.l_46
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_47
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11] <=> self.l_48
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_49
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_50
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_51
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14] <=> self.l_52
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu] <=> self.l_53
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1] <=> self.l_54
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2] <=> self.l_55
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] <=> self.l_56
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu] <=> self.l_57
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv] <=> self.l_58
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm] <=> self.l_59
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_60
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_61
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_62
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_63
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_64
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_65
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2] <=> self.l_66
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_67
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_68
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_69
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] <=> self.l_70
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] <=> self.l_71
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] <=> self.l_72
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] <=> self.l_73
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] <=> self.l_74
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] <=> self.l_75
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5] <=> self.l_76
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_77
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_78
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_79
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_80
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_81
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_82
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_83
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_84
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_85
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_86
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_87
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_88
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_89
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_90
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_91
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8] <=> self.l_92
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_93
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_94
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_95
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] <=> self.l_96
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] <=> self.l_97
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_98
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11] <=> self.l_99
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_100
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_101
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_102
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14] <=> self.l_103
        # input0 <=> x0

        # calling AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu] with arguments:
        # input0
        t_0 = self.l_0(x0)
        # calling AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/ReLU[relu]
        t_1 = self.l_1(t_0)
        # calling AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_2 = self.l_2(t_1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu] with arguments:
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_3 = self.l_3(t_2)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_4 = self.l_6(t_2)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu] with arguments:
        # AmoebaNet_D/Stem[stem]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_5 = self.l_53(t_2)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1460
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1461
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1462
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1463
        t_6 = t_5[0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/ReLU[relu]
        t_7 = self.l_4(t_3)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/ReLU[relu]
        t_8 = self.l_7(t_4)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/ReLU[relu]
        t_9 = self.l_54(t_5)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/aten::slice1464
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1465
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1466
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1467
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1468
        t_10 = t_6[:, 0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/Conv2d[conv]
        t_11 = self.l_5(t_7)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/Conv2d[conv]
        t_12 = self.l_8(t_8)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/aten::slice1469
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1470
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1471
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1472
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1473
        t_13 = t_10[:, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        t_14 = self.l_16(t_11)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        t_15 = self.l_49(t_11)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_16 = self.l_9(t_12)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_17 = self.l_12(t_12)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/aten::slice1474
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1475
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1476
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1477
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1478
        t_18 = t_13[:, :, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_19 = self.l_17(t_14)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        t_20 = self.l_50(t_15)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        t_21 = self.l_10(t_16)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        t_22 = self.l_13(t_17)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/aten::slice1479
        t_23 = self.l_55(t_18)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_24 = self.l_18(t_19)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_25 = self.l_51(t_20)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_26 = self.l_11(t_21)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_27 = self.l_14(t_22)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        t_28 = [t_9, t_23]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::ListConstruct1499
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/prim::Constant1500
        t_29 = torch.cat(t_28, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_30 = self.l_19(t_24)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_31 = torch.add(t_26, t_27)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/aten::cat1501
        t_32 = self.l_56(t_29)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        t_33 = self.l_20(t_30)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]
        t_34 = self.l_26(t_31)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]
        t_35 = self.l_38(t_31)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]
        t_36 = self.l_42(t_31)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_37 = self.l_67(t_32)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_38 = self.l_100(t_32)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        t_39 = self.l_21(t_33)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_40 = self.l_27(t_34)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        t_41 = self.l_39(t_35)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        t_42 = self.l_43(t_36)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_43 = self.l_68(t_37)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        t_44 = self.l_101(t_38)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        t_45 = self.l_22(t_39)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_46 = self.l_28(t_40)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_47 = self.l_40(t_41)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_48 = self.l_44(t_42)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_49 = self.l_69(t_43)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_50 = self.l_102(t_44)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        t_51 = self.l_23(t_45)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_52 = self.l_29(t_46)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_53 = self.l_70(t_49)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        t_54 = self.l_24(t_51)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[2]
        t_55 = torch.add(t_54, t_31)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_56 = self.l_30(t_52)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        t_57 = self.l_71(t_53)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_58 = torch.add(t_55, t_25)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[5]
        t_59 = self.l_45(t_55)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_60 = self.l_31(t_56)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        t_61 = self.l_72(t_57)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        t_62 = self.l_46(t_59)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_63 = self.l_32(t_60)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        t_64 = self.l_73(t_61)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        t_65 = self.l_47(t_62)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_66 = self.l_33(t_63)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        t_67 = torch.add(t_48, t_65)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        t_68 = self.l_74(t_64)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_69 = self.l_34(t_66)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        t_70 = self.l_75(t_68)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_71 = self.l_35(t_69)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_72 = self.l_36(t_71)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_73 = self.l_37(t_72)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_74 = torch.add(t_73, t_47)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[8]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/Sequential[layers]/MergeTwo[14]
        t_75 = [t_74, t_67, t_58]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/prim::ListConstruct1437
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/prim::Constant1438
        t_76 = torch.cat(t_75, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/aten::cat1439
        t_77 = self.l_57(t_76)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/ReLU[relu]
        t_78 = self.l_58(t_77)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/Conv2d[conv]
        t_79 = self.l_59(t_78)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_80 = self.l_60(t_79)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_81 = self.l_63(t_79)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        t_82 = self.l_61(t_80)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        t_83 = self.l_64(t_81)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_84 = self.l_62(t_82)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_85 = self.l_65(t_83)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_86 = torch.add(t_84, t_85)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]
        t_87 = torch.add(t_70, t_86)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_88 = torch.add(t_87, t_50)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]
        t_89 = self.l_77(t_86)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]
        t_90 = self.l_89(t_86)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[2]
        t_91 = self.l_93(t_86)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[5]
        t_92 = self.l_96(t_87)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_93 = self.l_78(t_89)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        t_94 = self.l_90(t_90)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        t_95 = self.l_94(t_91)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        t_96 = self.l_97(t_92)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_97 = self.l_79(t_93)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_98 = self.l_91(t_94)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_99 = self.l_95(t_95)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        t_100 = self.l_98(t_96)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_101 = self.l_80(t_97)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        t_102 = torch.add(t_99, t_100)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_103 = self.l_81(t_101)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_104 = self.l_82(t_103)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_105 = self.l_83(t_104)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_106 = self.l_84(t_105)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_107 = self.l_85(t_106)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_108 = self.l_86(t_107)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_109 = self.l_87(t_108)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_110 = self.l_88(t_109)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_111 = torch.add(t_110, t_98)
        # returing:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/aten::cat1439
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]
        return (t_76, t_102, t_88, t_111)

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
        result = dict()
        for k, v in state.items():
            if k in lookup:
                result[lookup[k]] = v
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                result[new_k] = v
        return result

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)


class AmoebaNet_DPartition1(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(AmoebaNet_DPartition1, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 108)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu] was expected but not given'
        self.l_0 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu]']
        assert isinstance(self.l_0,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_0)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1] was expected but not given'
        self.l_1 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1]']
        assert isinstance(self.l_1,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1]] is expected to be of type Conv2d but was of type {type(self.l_1)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2] was expected but not given'
        self.l_2 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2]']
        assert isinstance(self.l_2,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] was expected but not given'
        self.l_3 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]']
        assert isinstance(self.l_3,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu] was expected but not given'
        self.l_4 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu]']
        assert isinstance(self.l_4,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_4)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv] was expected but not given'
        self.l_5 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv]']
        assert isinstance(self.l_5,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_5)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm] was expected but not given'
        self.l_6 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]']
        assert isinstance(self.l_6,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_6)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_7 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_7,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_7)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_8 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_8,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_9 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_9,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_10 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_10,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_10)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_11 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_11,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_11)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_12 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_12,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_12)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2] was expected but not given'
        self.l_13 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2]']
        assert isinstance(self.l_13,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2]] is expected to be of type MergeTwo but was of type {type(self.l_13)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_14 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_14,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_14)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_15 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_15,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_15)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_16 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_16,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_16)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5] was expected but not given'
        self.l_17 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5]']
        assert isinstance(self.l_17,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5]] is expected to be of type MergeTwo but was of type {type(self.l_17)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_18 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_18,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_18)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_19 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_19,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_19)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_20 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_20,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_20)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_21 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_21,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_21)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_22 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_22,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_22)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_23 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_23,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_23)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_24 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_24,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_24)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_25 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_25,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_25)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_26 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_26,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_26)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_27 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_27,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_27)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_28 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_28,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_28)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_29 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_29,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_29)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8] was expected but not given'
        self.l_30 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8]']
        assert isinstance(self.l_30,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8]] is expected to be of type MergeTwo but was of type {type(self.l_30)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_31 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_31,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_31)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_32 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_32,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_32)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_33 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_33,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_33)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_34 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_34,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_34)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_35 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_35,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_35)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_36 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_36,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_36)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_37 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_37,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_37)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_38 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_38,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_38)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_39 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_39,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_39)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_40 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_40,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_40)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_41 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_41,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_41)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_42 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_42,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_42)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_43 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_43,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_43)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_44 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_44,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_44)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_45 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_45,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_45)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11] was expected but not given'
        self.l_46 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11]']
        assert isinstance(self.l_46,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11]] is expected to be of type MergeTwo but was of type {type(self.l_46)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_47 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_47,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_47)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_48 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_48,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_48)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_49 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_49,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_49)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_50 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_50,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_50)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_51 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_51,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_51)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_52 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_52,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_52)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14] was expected but not given'
        self.l_53 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14]']
        assert isinstance(self.l_53,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14]] is expected to be of type MergeTwo but was of type {type(self.l_53)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu] was expected but not given'
        self.l_54 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu]']
        assert isinstance(self.l_54,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_54)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv] was expected but not given'
        self.l_55 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv]']
        assert isinstance(self.l_55,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_55)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm] was expected but not given'
        self.l_56 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm]']
        assert isinstance(self.l_56,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_56)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu] was expected but not given'
        self.l_57 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu]']
        assert isinstance(self.l_57,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_57)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv] was expected but not given'
        self.l_58 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv]']
        assert isinstance(self.l_58,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_58)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm] was expected but not given'
        self.l_59 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm]']
        assert isinstance(self.l_59,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_59)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_60 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_60,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_60)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_61 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_61,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_61)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_62 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_62,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_62)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_63 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_63,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_63)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_64 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_64,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_64)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_65 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_65,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_65)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2] was expected but not given'
        self.l_66 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]']
        assert isinstance(self.l_66,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]] is expected to be of type MergeTwo but was of type {type(self.l_66)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_67 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_67,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_67)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_68 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_68,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_68)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_69 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_69,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_69)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] was expected but not given'
        self.l_70 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]']
        assert isinstance(self.l_70,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_70)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] was expected but not given'
        self.l_71 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]']
        assert isinstance(self.l_71,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_71)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] was expected but not given'
        self.l_72 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]']
        assert isinstance(self.l_72,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_72)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] was expected but not given'
        self.l_73 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]']
        assert isinstance(self.l_73,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_73)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] was expected but not given'
        self.l_74 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]']
        assert isinstance(self.l_74,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_74)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_75 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_75,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_75)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5] was expected but not given'
        self.l_76 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5]']
        assert isinstance(self.l_76,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5]] is expected to be of type MergeTwo but was of type {type(self.l_76)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_77 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_77,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_77)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_78 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_78,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_78)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_79 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_79,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_79)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_80 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_80,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_80)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_81 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_81,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_81)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_82 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_82,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_82)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_83 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_83,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_83)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_84 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_84,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_84)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_85 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_85,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_85)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_86 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_86,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_86)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_87 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_87,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_87)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_88 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_88,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_88)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_89 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_89,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_89)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_90 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_90,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_90)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_91 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_91,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_91)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8] was expected but not given'
        self.l_92 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8]']
        assert isinstance(self.l_92,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8]] is expected to be of type MergeTwo but was of type {type(self.l_92)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_93 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_93,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_93)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_94 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_94,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_94)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_95 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_95,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_95)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_96 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_96,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_96)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_97 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_97,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_97)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_98 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_98,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_98)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11] was expected but not given'
        self.l_99 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11]']
        assert isinstance(self.l_99,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11]] is expected to be of type MergeTwo but was of type {type(self.l_99)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_100 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_100,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_100)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_101 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_101,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_101)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_102 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_102,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_102)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14] was expected but not given'
        self.l_103 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14]']
        assert isinstance(self.l_103,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14]] is expected to be of type MergeTwo but was of type {type(self.l_103)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu] was expected but not given'
        self.l_104 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu]']
        assert isinstance(self.l_104,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_104)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1] was expected but not given'
        self.l_105 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1]']
        assert isinstance(self.l_105,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1]] is expected to be of type Conv2d but was of type {type(self.l_105)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2] was expected but not given'
        self.l_106 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2]']
        assert isinstance(self.l_106,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2]] is expected to be of type Conv2d but was of type {type(self.l_106)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] was expected but not given'
        self.l_107 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]']
        assert isinstance(self.l_107,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]] is expected to be of type BatchNorm2d but was of type {type(self.l_107)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.lookup = {'l_0': 'cells.2.preprocess0.relu', 'l_1': 'cells.2.preprocess0.conv_1', 'l_2': 'cells.2.preprocess0.conv_2', 'l_3': 'cells.2.preprocess0.bn', 'l_4': 'cells.2.preprocess1.relu', 'l_5': 'cells.2.preprocess1.conv', 'l_6': 'cells.2.preprocess1.norm', 'l_7': 'cells.2.layers.0.module.pool', 'l_8': 'cells.2.layers.0.module.conv_cell.conv', 'l_9': 'cells.2.layers.0.module.conv_cell.norm', 'l_10': 'cells.2.layers.1.module.relu', 'l_11': 'cells.2.layers.1.module.conv', 'l_12': 'cells.2.layers.1.module.norm', 'l_13': 'cells.2.layers.2', 'l_14': 'cells.2.layers.4.module.pool', 'l_15': 'cells.2.layers.4.module.conv_cell.conv', 'l_16': 'cells.2.layers.4.module.conv_cell.norm', 'l_17': 'cells.2.layers.5', 'l_18': 'cells.2.layers.7.module.conv1_1x1.relu', 'l_19': 'cells.2.layers.7.module.conv1_1x1.conv', 'l_20': 'cells.2.layers.7.module.conv1_1x1.norm', 'l_21': 'cells.2.layers.7.module.conv2_1x7.relu', 'l_22': 'cells.2.layers.7.module.conv2_1x7.conv', 'l_23': 'cells.2.layers.7.module.conv2_1x7.norm', 'l_24': 'cells.2.layers.7.module.conv3_7x1.relu', 'l_25': 'cells.2.layers.7.module.conv3_7x1.conv', 'l_26': 'cells.2.layers.7.module.conv3_7x1.norm', 'l_27': 'cells.2.layers.7.module.conv4_1x1.relu', 'l_28': 'cells.2.layers.7.module.conv4_1x1.conv', 'l_29': 'cells.2.layers.7.module.conv4_1x1.norm', 'l_30': 'cells.2.layers.8', 'l_31': 'cells.2.layers.9.module.relu', 'l_32': 'cells.2.layers.9.module.conv', 'l_33': 'cells.2.layers.9.module.norm', 'l_34': 'cells.2.layers.10.module.conv1_1x1.relu', 'l_35': 'cells.2.layers.10.module.conv1_1x1.conv', 'l_36': 'cells.2.layers.10.module.conv1_1x1.norm', 'l_37': 'cells.2.layers.10.module.conv2_1x7.relu', 'l_38': 'cells.2.layers.10.module.conv2_1x7.conv', 'l_39': 'cells.2.layers.10.module.conv2_1x7.norm', 'l_40': 'cells.2.layers.10.module.conv3_7x1.relu', 'l_41': 'cells.2.layers.10.module.conv3_7x1.conv', 'l_42': 'cells.2.layers.10.module.conv3_7x1.norm', 'l_43': 'cells.2.layers.10.module.conv4_1x1.relu', 'l_44': 'cells.2.layers.10.module.conv4_1x1.conv', 'l_45': 'cells.2.layers.10.module.conv4_1x1.norm', 'l_46': 'cells.2.layers.11', 'l_47': 'cells.2.layers.12.module.pool', 'l_48': 'cells.2.layers.12.module.conv_cell.conv', 'l_49': 'cells.2.layers.12.module.conv_cell.norm', 'l_50': 'cells.2.layers.13.module.relu', 'l_51': 'cells.2.layers.13.module.conv', 'l_52': 'cells.2.layers.13.module.norm', 'l_53': 'cells.2.layers.14', 'l_54': 'cells.3.preprocess0.relu', 'l_55': 'cells.3.preprocess0.conv', 'l_56': 'cells.3.preprocess0.norm', 'l_57': 'cells.3.preprocess1.relu', 'l_58': 'cells.3.preprocess1.conv', 'l_59': 'cells.3.preprocess1.norm', 'l_60': 'cells.3.layers.0.module.pool', 'l_61': 'cells.3.layers.0.module.conv_cell.conv', 'l_62': 'cells.3.layers.0.module.conv_cell.norm', 'l_63': 'cells.3.layers.1.module.pool', 'l_64': 'cells.3.layers.1.module.conv_cell.conv', 'l_65': 'cells.3.layers.1.module.conv_cell.norm', 'l_66': 'cells.3.layers.2', 'l_67': 'cells.3.layers.3.module.conv1_1x1.relu', 'l_68': 'cells.3.layers.3.module.conv1_1x1.conv', 'l_69': 'cells.3.layers.3.module.conv1_1x1.norm', 'l_70': 'cells.3.layers.3.module.conv2_3x3.relu', 'l_71': 'cells.3.layers.3.module.conv2_3x3.conv', 'l_72': 'cells.3.layers.3.module.conv2_3x3.norm', 'l_73': 'cells.3.layers.3.module.conv3_1x1.relu', 'l_74': 'cells.3.layers.3.module.conv3_1x1.conv', 'l_75': 'cells.3.layers.3.module.conv3_1x1.norm', 'l_76': 'cells.3.layers.5', 'l_77': 'cells.3.layers.6.module.conv1_1x1.relu', 'l_78': 'cells.3.layers.6.module.conv1_1x1.conv', 'l_79': 'cells.3.layers.6.module.conv1_1x1.norm', 'l_80': 'cells.3.layers.6.module.conv2_1x7.relu', 'l_81': 'cells.3.layers.6.module.conv2_1x7.conv', 'l_82': 'cells.3.layers.6.module.conv2_1x7.norm', 'l_83': 'cells.3.layers.6.module.conv3_7x1.relu', 'l_84': 'cells.3.layers.6.module.conv3_7x1.conv', 'l_85': 'cells.3.layers.6.module.conv3_7x1.norm', 'l_86': 'cells.3.layers.6.module.conv4_1x1.relu', 'l_87': 'cells.3.layers.6.module.conv4_1x1.conv', 'l_88': 'cells.3.layers.6.module.conv4_1x1.norm', 'l_89': 'cells.3.layers.7.module.pool', 'l_90': 'cells.3.layers.7.module.conv_cell.conv', 'l_91': 'cells.3.layers.7.module.conv_cell.norm', 'l_92': 'cells.3.layers.8', 'l_93': 'cells.3.layers.9.module.pool', 'l_94': 'cells.3.layers.9.module.conv_cell.conv', 'l_95': 'cells.3.layers.9.module.conv_cell.norm', 'l_96': 'cells.3.layers.10.module.relu', 'l_97': 'cells.3.layers.10.module.conv', 'l_98': 'cells.3.layers.10.module.norm', 'l_99': 'cells.3.layers.11', 'l_100': 'cells.3.layers.13.module.pool', 'l_101': 'cells.3.layers.13.module.conv_cell.conv', 'l_102': 'cells.3.layers.13.module.conv_cell.norm', 'l_103': 'cells.3.layers.14', 'l_104': 'cells.4.preprocess0.relu', 'l_105': 'cells.4.preprocess0.conv_1', 'l_106': 'cells.4.preprocess0.conv_2', 'l_107': 'cells.4.preprocess0.bn'}

    def forward(self, x0, x1, x2, x3):
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu] <=> self.l_0
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1] <=> self.l_1
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2] <=> self.l_2
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] <=> self.l_3
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu] <=> self.l_4
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv] <=> self.l_5
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm] <=> self.l_6
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_7
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_8
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_9
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] <=> self.l_10
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] <=> self.l_11
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_12
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2] <=> self.l_13
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_14
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_15
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_16
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5] <=> self.l_17
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_18
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_19
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_20
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_21
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_22
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_23
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_24
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_25
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_26
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_27
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_28
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_29
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8] <=> self.l_30
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] <=> self.l_31
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] <=> self.l_32
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_33
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_34
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_35
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_36
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_37
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_38
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_39
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_40
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_41
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_42
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_43
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_44
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_45
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11] <=> self.l_46
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_47
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_48
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_49
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] <=> self.l_50
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] <=> self.l_51
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_52
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14] <=> self.l_53
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu] <=> self.l_54
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv] <=> self.l_55
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm] <=> self.l_56
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu] <=> self.l_57
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv] <=> self.l_58
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm] <=> self.l_59
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_60
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_61
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_62
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_63
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_64
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_65
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2] <=> self.l_66
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_67
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_68
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_69
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] <=> self.l_70
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] <=> self.l_71
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] <=> self.l_72
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] <=> self.l_73
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] <=> self.l_74
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] <=> self.l_75
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5] <=> self.l_76
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_77
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_78
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_79
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_80
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_81
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_82
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_83
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_84
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_85
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_86
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_87
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_88
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_89
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_90
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_91
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8] <=> self.l_92
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_93
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_94
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_95
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] <=> self.l_96
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] <=> self.l_97
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_98
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11] <=> self.l_99
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_100
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_101
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_102
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14] <=> self.l_103
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu] <=> self.l_104
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1] <=> self.l_105
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2] <=> self.l_106
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] <=> self.l_107
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/aten::cat1439 <=> x0
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11] <=> x1
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14] <=> x2
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8] <=> x3

        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[0]/aten::cat1439
        t_0 = self.l_0(x0)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[8]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/Sequential[layers]/MergeTwo[14]
        t_1 = [x3, x1, x2]
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2237
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2238
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2239
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2240
        t_2 = t_0[0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/ReLU[relu]
        t_3 = self.l_1(t_0)
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/prim::ListConstruct2214
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/prim::Constant2215
        t_4 = torch.cat(t_1, 1)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/aten::slice2241
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2242
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2243
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2244
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2245
        t_5 = t_2[:, 0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/aten::cat2216
        t_6 = self.l_4(t_4)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[1]/aten::cat2216
        t_7 = self.l_54(t_4)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/aten::slice2246
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2247
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2248
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2249
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2250
        t_8 = t_5[:, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/ReLU[relu]
        t_9 = self.l_5(t_6)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/ReLU[relu]
        t_10 = self.l_55(t_7)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/aten::slice2251
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2252
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2253
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2254
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2255
        t_11 = t_8[:, :, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/Conv2d[conv]
        t_12 = self.l_6(t_9)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/Conv2d[conv]
        t_13 = self.l_56(t_10)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/aten::slice2256
        t_14 = self.l_2(t_11)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        t_15 = [t_3, t_14]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_16 = self.l_18(t_12)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_17 = self.l_31(t_12)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_18 = self.l_34(t_12)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        t_19 = self.l_67(t_13)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        t_20 = self.l_100(t_13)
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::ListConstruct2276
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/prim::Constant2277
        t_21 = torch.cat(t_15, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_22 = self.l_19(t_16)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]
        t_23 = self.l_32(t_17)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_24 = self.l_35(t_18)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_25 = self.l_68(t_19)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        t_26 = self.l_101(t_20)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/aten::cat2278
        t_27 = self.l_3(t_21)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_28 = self.l_20(t_22)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]
        t_29 = self.l_33(t_23)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_30 = self.l_36(t_24)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_31 = self.l_69(t_25)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_32 = self.l_102(t_26)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_33 = self.l_7(t_27)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_34 = self.l_10(t_27)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_35 = self.l_47(t_27)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_36 = self.l_21(t_28)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_37 = self.l_37(t_30)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_38 = self.l_70(t_31)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]
        t_39 = self.l_8(t_33)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]
        t_40 = self.l_11(t_34)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]
        t_41 = self.l_48(t_35)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_42 = self.l_22(t_36)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_43 = self.l_38(t_37)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        t_44 = self.l_71(t_38)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_45 = self.l_9(t_39)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]
        t_46 = self.l_12(t_40)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_47 = self.l_49(t_41)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_48 = self.l_23(t_42)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_49 = self.l_39(t_43)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        t_50 = self.l_72(t_44)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]
        t_51 = torch.add(t_45, t_46)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_52 = self.l_24(t_48)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_53 = self.l_40(t_49)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        t_54 = self.l_73(t_50)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2]
        t_55 = self.l_14(t_51)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_56 = self.l_25(t_52)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_57 = self.l_41(t_53)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        t_58 = self.l_74(t_54)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]
        t_59 = self.l_15(t_55)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_60 = self.l_26(t_56)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_61 = self.l_42(t_57)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        t_62 = self.l_75(t_58)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_63 = self.l_16(t_59)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[2]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_64 = torch.add(t_51, t_63)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_65 = self.l_27(t_60)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_66 = self.l_43(t_61)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[5]
        t_67 = self.l_50(t_64)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_68 = self.l_28(t_65)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_69 = self.l_44(t_66)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]
        t_70 = self.l_51(t_67)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_71 = self.l_29(t_68)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_72 = self.l_45(t_69)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]
        t_73 = self.l_52(t_70)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        t_74 = torch.add(t_27, t_71)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]
        t_75 = torch.add(t_47, t_73)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        t_76 = torch.add(t_29, t_72)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[8]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/Sequential[layers]/MergeTwo[14]
        t_77 = [t_74, t_76, t_75]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/prim::ListConstruct3009
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/prim::Constant3010
        t_78 = torch.cat(t_77, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/aten::cat3011
        t_79 = self.l_57(t_78)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[2]/aten::cat3011
        t_80 = self.l_104(t_78)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3767
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3768
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3769
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3770
        t_81 = t_80[0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/ReLU[relu]
        t_82 = self.l_58(t_79)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/ReLU[relu]
        t_83 = self.l_105(t_80)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/aten::slice3771
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3772
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3773
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3774
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3775
        t_84 = t_81[:, 0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/Conv2d[conv]
        t_85 = self.l_59(t_82)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/aten::slice3776
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3777
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3778
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3779
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3780
        t_86 = t_84[:, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_87 = self.l_60(t_85)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_88 = self.l_63(t_85)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/aten::slice3781
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3782
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3783
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3784
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3785
        t_89 = t_86[:, :, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        t_90 = self.l_61(t_87)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        t_91 = self.l_64(t_88)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/aten::slice3786
        t_92 = self.l_106(t_89)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_93 = self.l_62(t_90)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_94 = self.l_65(t_91)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        t_95 = [t_83, t_92]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::ListConstruct3806
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/prim::Constant3807
        t_96 = torch.cat(t_95, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_97 = torch.add(t_93, t_94)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]
        t_98 = torch.add(t_62, t_97)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_99 = torch.add(t_98, t_32)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/aten::cat3808
        t_100 = self.l_107(t_96)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]
        t_101 = self.l_77(t_97)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]
        t_102 = self.l_89(t_97)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[2]
        t_103 = self.l_93(t_97)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[5]
        t_104 = self.l_96(t_98)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_105 = self.l_78(t_101)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        t_106 = self.l_90(t_102)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        t_107 = self.l_94(t_103)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        t_108 = self.l_97(t_104)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_109 = self.l_79(t_105)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_110 = self.l_91(t_106)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_111 = self.l_95(t_107)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        t_112 = self.l_98(t_108)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_113 = self.l_80(t_109)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        t_114 = torch.add(t_111, t_112)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_115 = self.l_81(t_113)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_116 = self.l_82(t_115)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_117 = self.l_83(t_116)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_118 = self.l_84(t_117)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_119 = self.l_85(t_118)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_120 = self.l_86(t_119)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_121 = self.l_87(t_120)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_122 = self.l_88(t_121)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_123 = torch.add(t_122, t_110)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[8]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/Sequential[layers]/MergeTwo[14]
        t_124 = [t_123, t_114, t_99]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/prim::ListConstruct3744
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/prim::Constant3745
        t_125 = torch.cat(t_124, 1)
        # returing:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/aten::cat3746
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        return (t_125, t_100)

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
        result = dict()
        for k, v in state.items():
            if k in lookup:
                result[lookup[k]] = v
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                result[new_k] = v
        return result

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)


class AmoebaNet_DPartition2(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(AmoebaNet_DPartition2, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 156)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu] was expected but not given'
        self.l_0 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu]']
        assert isinstance(self.l_0,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_0)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv] was expected but not given'
        self.l_1 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv]']
        assert isinstance(self.l_1,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_1)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm] was expected but not given'
        self.l_2 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]']
        assert isinstance(self.l_2,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_2)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_3 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_3,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_3)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_4 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_4,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_5 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_5,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_6 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_6,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_6)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_7 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_7,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_7)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_8 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_8,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_8)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2] was expected but not given'
        self.l_9 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2]']
        assert isinstance(self.l_9,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2]] is expected to be of type MergeTwo but was of type {type(self.l_9)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_10 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_10,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_10)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_11 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_11,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_11)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_12 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_12,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_12)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5] was expected but not given'
        self.l_13 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5]']
        assert isinstance(self.l_13,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5]] is expected to be of type MergeTwo but was of type {type(self.l_13)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_14 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_14,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_14)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_15 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_15,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_15)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_16 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_16,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_16)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_17 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_17,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_17)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_18 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_18,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_18)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_19 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_19,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_19)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_20 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_20,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_20)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_21 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_21,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_21)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_22 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_22,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_22)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_23 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_23,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_23)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_24 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_24,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_24)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_25 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_25,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_25)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8] was expected but not given'
        self.l_26 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8]']
        assert isinstance(self.l_26,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8]] is expected to be of type MergeTwo but was of type {type(self.l_26)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_27 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_27,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_27)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_28 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_28,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_28)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_29 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_29,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_29)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_30 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_30,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_30)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_31 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_31,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_31)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_32 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_32,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_32)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_33 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_33,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_33)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_34 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_34,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_34)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_35 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_35,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_35)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_36 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_36,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_36)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_37 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_37,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_37)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_38 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_38,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_38)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_39 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_39,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_39)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_40 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_40,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_40)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_41 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_41,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_41)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11] was expected but not given'
        self.l_42 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11]']
        assert isinstance(self.l_42,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11]] is expected to be of type MergeTwo but was of type {type(self.l_42)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_43 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_43,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_43)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_44 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_44,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_44)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_45 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_45,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_45)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_46 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_46,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_46)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_47 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_47,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_47)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_48 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_48,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_48)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14] was expected but not given'
        self.l_49 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14]']
        assert isinstance(self.l_49,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14]] is expected to be of type MergeTwo but was of type {type(self.l_49)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu] was expected but not given'
        self.l_50 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu]']
        assert isinstance(self.l_50,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_50)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv] was expected but not given'
        self.l_51 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv]']
        assert isinstance(self.l_51,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_51)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm] was expected but not given'
        self.l_52 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm]']
        assert isinstance(self.l_52,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_52)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu] was expected but not given'
        self.l_53 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu]']
        assert isinstance(self.l_53,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_53)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv] was expected but not given'
        self.l_54 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv]']
        assert isinstance(self.l_54,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_54)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm] was expected but not given'
        self.l_55 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm]']
        assert isinstance(self.l_55,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_55)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_56 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_56,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_56)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_57 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_57,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_57)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_58 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_58,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_58)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_59 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_59,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_59)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_60 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_60,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_60)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_61 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_61,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_61)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2] was expected but not given'
        self.l_62 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]']
        assert isinstance(self.l_62,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]] is expected to be of type MergeTwo but was of type {type(self.l_62)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_63 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_63,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_63)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_64 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_64,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_64)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_65 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_65,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_65)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] was expected but not given'
        self.l_66 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]']
        assert isinstance(self.l_66,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_66)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] was expected but not given'
        self.l_67 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]']
        assert isinstance(self.l_67,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_67)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] was expected but not given'
        self.l_68 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]']
        assert isinstance(self.l_68,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_68)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] was expected but not given'
        self.l_69 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]']
        assert isinstance(self.l_69,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_69)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] was expected but not given'
        self.l_70 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]']
        assert isinstance(self.l_70,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_70)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_71 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_71,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_71)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5] was expected but not given'
        self.l_72 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5]']
        assert isinstance(self.l_72,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5]] is expected to be of type MergeTwo but was of type {type(self.l_72)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_73 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_73,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_73)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_74 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_74,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_74)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_75 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_75,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_75)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_76 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_76,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_76)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_77 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_77,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_77)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_78 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_78,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_78)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_79 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_79,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_79)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_80 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_80,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_80)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_81 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_81,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_81)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_82 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_82,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_82)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_83 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_83,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_83)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_84 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_84,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_84)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_85 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_85,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_85)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_86 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_86,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_86)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_87 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_87,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_87)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8] was expected but not given'
        self.l_88 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8]']
        assert isinstance(self.l_88,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8]] is expected to be of type MergeTwo but was of type {type(self.l_88)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_89 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_89,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_89)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_90 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_90,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_90)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_91 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_91,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_91)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_92 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_92,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_92)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_93 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_93,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_93)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_94 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_94,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_94)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11] was expected but not given'
        self.l_95 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11]']
        assert isinstance(self.l_95,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11]] is expected to be of type MergeTwo but was of type {type(self.l_95)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] was expected but not given'
        self.l_96 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]']
        assert isinstance(self.l_96,MaxPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]] is expected to be of type MaxPool2d but was of type {type(self.l_96)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_97 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_97,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_97)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_98 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_98,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_98)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14] was expected but not given'
        self.l_99 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14]']
        assert isinstance(self.l_99,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14]] is expected to be of type MergeTwo but was of type {type(self.l_99)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu] was expected but not given'
        self.l_100 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu]']
        assert isinstance(self.l_100,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_100)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1] was expected but not given'
        self.l_101 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1]']
        assert isinstance(self.l_101,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1]] is expected to be of type Conv2d but was of type {type(self.l_101)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2] was expected but not given'
        self.l_102 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2]']
        assert isinstance(self.l_102,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2]] is expected to be of type Conv2d but was of type {type(self.l_102)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] was expected but not given'
        self.l_103 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]']
        assert isinstance(self.l_103,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]] is expected to be of type BatchNorm2d but was of type {type(self.l_103)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu] was expected but not given'
        self.l_104 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu]']
        assert isinstance(self.l_104,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_104)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv] was expected but not given'
        self.l_105 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv]']
        assert isinstance(self.l_105,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_105)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm] was expected but not given'
        self.l_106 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]']
        assert isinstance(self.l_106,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_106)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_107 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_107,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_107)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_108 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_108,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_108)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_109 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_109,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_109)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_110 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_110,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_110)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_111 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_111,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_111)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_112 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_112,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_112)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2] was expected but not given'
        self.l_113 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2]']
        assert isinstance(self.l_113,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2]] is expected to be of type MergeTwo but was of type {type(self.l_113)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_114 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_114,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_114)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_115 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_115,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_115)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_116 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_116,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_116)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5] was expected but not given'
        self.l_117 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5]']
        assert isinstance(self.l_117,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5]] is expected to be of type MergeTwo but was of type {type(self.l_117)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_118 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_118,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_118)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_119 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_119,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_119)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_120 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_120,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_120)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_121 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_121,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_121)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_122 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_122,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_122)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_123 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_123,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_123)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_124 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_124,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_124)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_125 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_125,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_125)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_126 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_126,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_126)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_127 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_127,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_127)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_128 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_128,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_128)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_129 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_129,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_129)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8] was expected but not given'
        self.l_130 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8]']
        assert isinstance(self.l_130,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8]] is expected to be of type MergeTwo but was of type {type(self.l_130)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_131 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_131,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_131)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_132 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_132,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_132)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_133 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_133,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_133)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] was expected but not given'
        self.l_134 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]']
        assert isinstance(self.l_134,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_134)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] was expected but not given'
        self.l_135 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]']
        assert isinstance(self.l_135,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_135)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_136 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_136,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_136)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] was expected but not given'
        self.l_137 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]']
        assert isinstance(self.l_137,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_137)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] was expected but not given'
        self.l_138 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]']
        assert isinstance(self.l_138,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_138)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] was expected but not given'
        self.l_139 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]']
        assert isinstance(self.l_139,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_139)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] was expected but not given'
        self.l_140 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]']
        assert isinstance(self.l_140,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_140)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] was expected but not given'
        self.l_141 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]']
        assert isinstance(self.l_141,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_141)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] was expected but not given'
        self.l_142 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_142,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_142)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] was expected but not given'
        self.l_143 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]']
        assert isinstance(self.l_143,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_143)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] was expected but not given'
        self.l_144 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]']
        assert isinstance(self.l_144,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_144)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] was expected but not given'
        self.l_145 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]']
        assert isinstance(self.l_145,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_145)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11] was expected but not given'
        self.l_146 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11]']
        assert isinstance(self.l_146,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11]] is expected to be of type MergeTwo but was of type {type(self.l_146)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] was expected but not given'
        self.l_147 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]']
        assert isinstance(self.l_147,AvgPool2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]] is expected to be of type AvgPool2d but was of type {type(self.l_147)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] was expected but not given'
        self.l_148 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]']
        assert isinstance(self.l_148,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_148)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] was expected but not given'
        self.l_149 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]']
        assert isinstance(self.l_149,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_149)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] was expected but not given'
        self.l_150 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]']
        assert isinstance(self.l_150,ReLU) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_150)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] was expected but not given'
        self.l_151 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]']
        assert isinstance(self.l_151,Conv2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]] is expected to be of type Conv2d but was of type {type(self.l_151)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] was expected but not given'
        self.l_152 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]']
        assert isinstance(self.l_152,BatchNorm2d) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]] is expected to be of type BatchNorm2d but was of type {type(self.l_152)}'
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14]
        assert 'AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14]' in layers, 'layer AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14] was expected but not given'
        self.l_153 = layers['AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14]']
        assert isinstance(self.l_153,MergeTwo) ,f'layers[AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14]] is expected to be of type MergeTwo but was of type {type(self.l_153)}'
        # AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling]
        assert 'AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling]' in layers, 'layer AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling] was expected but not given'
        self.l_154 = layers['AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling]']
        assert isinstance(self.l_154,AvgPool2d) ,f'layers[AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling]] is expected to be of type AvgPool2d but was of type {type(self.l_154)}'
        # AmoebaNet_D/Classifier[classifier]/Linear[classifier]
        assert 'AmoebaNet_D/Classifier[classifier]/Linear[classifier]' in layers, 'layer AmoebaNet_D/Classifier[classifier]/Linear[classifier] was expected but not given'
        self.l_155 = layers['AmoebaNet_D/Classifier[classifier]/Linear[classifier]']
        assert isinstance(self.l_155,Linear) ,f'layers[AmoebaNet_D/Classifier[classifier]/Linear[classifier]] is expected to be of type Linear but was of type {type(self.l_155)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.lookup = {'l_0': 'cells.4.preprocess1.relu', 'l_1': 'cells.4.preprocess1.conv', 'l_2': 'cells.4.preprocess1.norm', 'l_3': 'cells.4.layers.0.module.pool', 'l_4': 'cells.4.layers.0.module.conv_cell.conv', 'l_5': 'cells.4.layers.0.module.conv_cell.norm', 'l_6': 'cells.4.layers.1.module.relu', 'l_7': 'cells.4.layers.1.module.conv', 'l_8': 'cells.4.layers.1.module.norm', 'l_9': 'cells.4.layers.2', 'l_10': 'cells.4.layers.4.module.pool', 'l_11': 'cells.4.layers.4.module.conv_cell.conv', 'l_12': 'cells.4.layers.4.module.conv_cell.norm', 'l_13': 'cells.4.layers.5', 'l_14': 'cells.4.layers.7.module.conv1_1x1.relu', 'l_15': 'cells.4.layers.7.module.conv1_1x1.conv', 'l_16': 'cells.4.layers.7.module.conv1_1x1.norm', 'l_17': 'cells.4.layers.7.module.conv2_1x7.relu', 'l_18': 'cells.4.layers.7.module.conv2_1x7.conv', 'l_19': 'cells.4.layers.7.module.conv2_1x7.norm', 'l_20': 'cells.4.layers.7.module.conv3_7x1.relu', 'l_21': 'cells.4.layers.7.module.conv3_7x1.conv', 'l_22': 'cells.4.layers.7.module.conv3_7x1.norm', 'l_23': 'cells.4.layers.7.module.conv4_1x1.relu', 'l_24': 'cells.4.layers.7.module.conv4_1x1.conv', 'l_25': 'cells.4.layers.7.module.conv4_1x1.norm', 'l_26': 'cells.4.layers.8', 'l_27': 'cells.4.layers.9.module.relu', 'l_28': 'cells.4.layers.9.module.conv', 'l_29': 'cells.4.layers.9.module.norm', 'l_30': 'cells.4.layers.10.module.conv1_1x1.relu', 'l_31': 'cells.4.layers.10.module.conv1_1x1.conv', 'l_32': 'cells.4.layers.10.module.conv1_1x1.norm', 'l_33': 'cells.4.layers.10.module.conv2_1x7.relu', 'l_34': 'cells.4.layers.10.module.conv2_1x7.conv', 'l_35': 'cells.4.layers.10.module.conv2_1x7.norm', 'l_36': 'cells.4.layers.10.module.conv3_7x1.relu', 'l_37': 'cells.4.layers.10.module.conv3_7x1.conv', 'l_38': 'cells.4.layers.10.module.conv3_7x1.norm', 'l_39': 'cells.4.layers.10.module.conv4_1x1.relu', 'l_40': 'cells.4.layers.10.module.conv4_1x1.conv', 'l_41': 'cells.4.layers.10.module.conv4_1x1.norm', 'l_42': 'cells.4.layers.11', 'l_43': 'cells.4.layers.12.module.pool', 'l_44': 'cells.4.layers.12.module.conv_cell.conv', 'l_45': 'cells.4.layers.12.module.conv_cell.norm', 'l_46': 'cells.4.layers.13.module.relu', 'l_47': 'cells.4.layers.13.module.conv', 'l_48': 'cells.4.layers.13.module.norm', 'l_49': 'cells.4.layers.14', 'l_50': 'cells.5.preprocess0.relu', 'l_51': 'cells.5.preprocess0.conv', 'l_52': 'cells.5.preprocess0.norm', 'l_53': 'cells.5.preprocess1.relu', 'l_54': 'cells.5.preprocess1.conv', 'l_55': 'cells.5.preprocess1.norm', 'l_56': 'cells.5.layers.0.module.pool', 'l_57': 'cells.5.layers.0.module.conv_cell.conv', 'l_58': 'cells.5.layers.0.module.conv_cell.norm', 'l_59': 'cells.5.layers.1.module.pool', 'l_60': 'cells.5.layers.1.module.conv_cell.conv', 'l_61': 'cells.5.layers.1.module.conv_cell.norm', 'l_62': 'cells.5.layers.2', 'l_63': 'cells.5.layers.3.module.conv1_1x1.relu', 'l_64': 'cells.5.layers.3.module.conv1_1x1.conv', 'l_65': 'cells.5.layers.3.module.conv1_1x1.norm', 'l_66': 'cells.5.layers.3.module.conv2_3x3.relu', 'l_67': 'cells.5.layers.3.module.conv2_3x3.conv', 'l_68': 'cells.5.layers.3.module.conv2_3x3.norm', 'l_69': 'cells.5.layers.3.module.conv3_1x1.relu', 'l_70': 'cells.5.layers.3.module.conv3_1x1.conv', 'l_71': 'cells.5.layers.3.module.conv3_1x1.norm', 'l_72': 'cells.5.layers.5', 'l_73': 'cells.5.layers.6.module.conv1_1x1.relu', 'l_74': 'cells.5.layers.6.module.conv1_1x1.conv', 'l_75': 'cells.5.layers.6.module.conv1_1x1.norm', 'l_76': 'cells.5.layers.6.module.conv2_1x7.relu', 'l_77': 'cells.5.layers.6.module.conv2_1x7.conv', 'l_78': 'cells.5.layers.6.module.conv2_1x7.norm', 'l_79': 'cells.5.layers.6.module.conv3_7x1.relu', 'l_80': 'cells.5.layers.6.module.conv3_7x1.conv', 'l_81': 'cells.5.layers.6.module.conv3_7x1.norm', 'l_82': 'cells.5.layers.6.module.conv4_1x1.relu', 'l_83': 'cells.5.layers.6.module.conv4_1x1.conv', 'l_84': 'cells.5.layers.6.module.conv4_1x1.norm', 'l_85': 'cells.5.layers.7.module.pool', 'l_86': 'cells.5.layers.7.module.conv_cell.conv', 'l_87': 'cells.5.layers.7.module.conv_cell.norm', 'l_88': 'cells.5.layers.8', 'l_89': 'cells.5.layers.9.module.pool', 'l_90': 'cells.5.layers.9.module.conv_cell.conv', 'l_91': 'cells.5.layers.9.module.conv_cell.norm', 'l_92': 'cells.5.layers.10.module.relu', 'l_93': 'cells.5.layers.10.module.conv', 'l_94': 'cells.5.layers.10.module.norm', 'l_95': 'cells.5.layers.11', 'l_96': 'cells.5.layers.13.module.pool', 'l_97': 'cells.5.layers.13.module.conv_cell.conv', 'l_98': 'cells.5.layers.13.module.conv_cell.norm', 'l_99': 'cells.5.layers.14', 'l_100': 'cells.6.preprocess0.relu', 'l_101': 'cells.6.preprocess0.conv_1', 'l_102': 'cells.6.preprocess0.conv_2', 'l_103': 'cells.6.preprocess0.bn', 'l_104': 'cells.6.preprocess1.relu', 'l_105': 'cells.6.preprocess1.conv', 'l_106': 'cells.6.preprocess1.norm', 'l_107': 'cells.6.layers.0.module.pool', 'l_108': 'cells.6.layers.0.module.conv_cell.conv', 'l_109': 'cells.6.layers.0.module.conv_cell.norm', 'l_110': 'cells.6.layers.1.module.relu', 'l_111': 'cells.6.layers.1.module.conv', 'l_112': 'cells.6.layers.1.module.norm', 'l_113': 'cells.6.layers.2', 'l_114': 'cells.6.layers.4.module.pool', 'l_115': 'cells.6.layers.4.module.conv_cell.conv', 'l_116': 'cells.6.layers.4.module.conv_cell.norm', 'l_117': 'cells.6.layers.5', 'l_118': 'cells.6.layers.7.module.conv1_1x1.relu', 'l_119': 'cells.6.layers.7.module.conv1_1x1.conv', 'l_120': 'cells.6.layers.7.module.conv1_1x1.norm', 'l_121': 'cells.6.layers.7.module.conv2_1x7.relu', 'l_122': 'cells.6.layers.7.module.conv2_1x7.conv', 'l_123': 'cells.6.layers.7.module.conv2_1x7.norm', 'l_124': 'cells.6.layers.7.module.conv3_7x1.relu', 'l_125': 'cells.6.layers.7.module.conv3_7x1.conv', 'l_126': 'cells.6.layers.7.module.conv3_7x1.norm', 'l_127': 'cells.6.layers.7.module.conv4_1x1.relu', 'l_128': 'cells.6.layers.7.module.conv4_1x1.conv', 'l_129': 'cells.6.layers.7.module.conv4_1x1.norm', 'l_130': 'cells.6.layers.8', 'l_131': 'cells.6.layers.9.module.relu', 'l_132': 'cells.6.layers.9.module.conv', 'l_133': 'cells.6.layers.9.module.norm', 'l_134': 'cells.6.layers.10.module.conv1_1x1.relu', 'l_135': 'cells.6.layers.10.module.conv1_1x1.conv', 'l_136': 'cells.6.layers.10.module.conv1_1x1.norm', 'l_137': 'cells.6.layers.10.module.conv2_1x7.relu', 'l_138': 'cells.6.layers.10.module.conv2_1x7.conv', 'l_139': 'cells.6.layers.10.module.conv2_1x7.norm', 'l_140': 'cells.6.layers.10.module.conv3_7x1.relu', 'l_141': 'cells.6.layers.10.module.conv3_7x1.conv', 'l_142': 'cells.6.layers.10.module.conv3_7x1.norm', 'l_143': 'cells.6.layers.10.module.conv4_1x1.relu', 'l_144': 'cells.6.layers.10.module.conv4_1x1.conv', 'l_145': 'cells.6.layers.10.module.conv4_1x1.norm', 'l_146': 'cells.6.layers.11', 'l_147': 'cells.6.layers.12.module.pool', 'l_148': 'cells.6.layers.12.module.conv_cell.conv', 'l_149': 'cells.6.layers.12.module.conv_cell.norm', 'l_150': 'cells.6.layers.13.module.relu', 'l_151': 'cells.6.layers.13.module.conv', 'l_152': 'cells.6.layers.13.module.norm', 'l_153': 'cells.6.layers.14', 'l_154': 'classifier.global_pooling', 'l_155': 'classifier.classifier'}

    def forward(self, x0, x1):
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu] <=> self.l_0
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv] <=> self.l_1
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm] <=> self.l_2
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_3
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_4
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_5
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] <=> self.l_6
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] <=> self.l_7
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_8
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2] <=> self.l_9
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_10
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_11
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_12
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5] <=> self.l_13
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_14
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_15
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_16
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_17
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_18
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_19
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_20
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_21
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_22
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_23
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_24
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_25
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8] <=> self.l_26
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] <=> self.l_27
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] <=> self.l_28
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_29
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_30
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_31
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_32
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_33
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_34
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_35
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_36
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_37
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_38
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_39
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_40
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_41
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11] <=> self.l_42
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_43
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_44
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_45
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] <=> self.l_46
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] <=> self.l_47
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_48
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14] <=> self.l_49
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu] <=> self.l_50
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv] <=> self.l_51
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm] <=> self.l_52
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu] <=> self.l_53
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv] <=> self.l_54
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm] <=> self.l_55
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_56
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_57
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_58
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_59
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_60
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_61
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2] <=> self.l_62
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_63
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_64
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_65
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] <=> self.l_66
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] <=> self.l_67
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] <=> self.l_68
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] <=> self.l_69
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] <=> self.l_70
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] <=> self.l_71
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5] <=> self.l_72
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_73
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_74
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_75
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_76
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_77
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_78
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_79
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_80
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_81
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_82
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_83
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_84
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_85
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_86
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_87
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8] <=> self.l_88
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_89
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_90
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_91
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] <=> self.l_92
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] <=> self.l_93
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_94
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11] <=> self.l_95
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] <=> self.l_96
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_97
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_98
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14] <=> self.l_99
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu] <=> self.l_100
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1] <=> self.l_101
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2] <=> self.l_102
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] <=> self.l_103
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu] <=> self.l_104
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv] <=> self.l_105
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm] <=> self.l_106
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_107
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_108
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_109
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] <=> self.l_110
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] <=> self.l_111
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_112
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2] <=> self.l_113
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_114
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_115
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_116
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5] <=> self.l_117
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_118
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_119
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_120
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_121
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_122
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_123
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_124
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_125
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_126
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_127
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_128
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_129
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8] <=> self.l_130
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] <=> self.l_131
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] <=> self.l_132
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_133
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] <=> self.l_134
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] <=> self.l_135
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] <=> self.l_136
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] <=> self.l_137
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] <=> self.l_138
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] <=> self.l_139
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] <=> self.l_140
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] <=> self.l_141
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] <=> self.l_142
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] <=> self.l_143
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] <=> self.l_144
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] <=> self.l_145
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11] <=> self.l_146
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] <=> self.l_147
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] <=> self.l_148
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] <=> self.l_149
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] <=> self.l_150
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] <=> self.l_151
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] <=> self.l_152
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14] <=> self.l_153
        # AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling] <=> self.l_154
        # AmoebaNet_D/Classifier[classifier]/Linear[classifier] <=> self.l_155
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/aten::cat3746 <=> x0
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] <=> x1

        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/aten::cat3746
        t_0 = self.l_50(x0)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_1 = self.l_43(x1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_2 = self.l_6(x1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_3 = self.l_3(x1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[3]/aten::cat3746
        t_4 = self.l_0(x0)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/ReLU[relu]
        t_5 = self.l_51(t_0)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]
        t_6 = self.l_44(t_1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]
        t_7 = self.l_7(t_2)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]
        t_8 = self.l_4(t_3)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/ReLU[relu]
        t_9 = self.l_1(t_4)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/Conv2d[conv]
        t_10 = self.l_52(t_5)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_11 = self.l_45(t_6)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]
        t_12 = self.l_8(t_7)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_13 = self.l_5(t_8)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/Conv2d[conv]
        t_14 = self.l_2(t_9)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        t_15 = self.l_63(t_10)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess0]/BatchNorm2d[norm]
        t_16 = self.l_96(t_10)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]
        t_17 = torch.add(t_13, t_12)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_18 = self.l_14(t_14)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_19 = self.l_27(t_14)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_20 = self.l_30(t_14)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_21 = self.l_64(t_15)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/MaxPool2d[pool]
        t_22 = self.l_97(t_16)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2]
        t_23 = self.l_10(t_17)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_24 = self.l_15(t_18)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]
        t_25 = self.l_28(t_19)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_26 = self.l_31(t_20)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_27 = self.l_65(t_21)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_28 = self.l_98(t_22)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]
        t_29 = self.l_11(t_23)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_30 = self.l_16(t_24)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]
        t_31 = self.l_29(t_25)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_32 = self.l_32(t_26)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_33 = self.l_66(t_27)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_34 = self.l_12(t_29)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[2]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_35 = torch.add(t_17, t_34)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_36 = self.l_17(t_30)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_37 = self.l_33(t_32)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/ReLU[relu]
        t_38 = self.l_67(t_33)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[5]
        t_39 = self.l_46(t_35)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_40 = self.l_18(t_36)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_41 = self.l_34(t_37)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/Conv2d[conv]
        t_42 = self.l_68(t_38)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]
        t_43 = self.l_47(t_39)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_44 = self.l_19(t_40)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_45 = self.l_35(t_41)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv2_3x3]/BatchNorm2d[norm]
        t_46 = self.l_69(t_42)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]
        t_47 = self.l_48(t_43)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_48 = self.l_20(t_44)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_49 = self.l_36(t_45)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/ReLU[relu]
        t_50 = self.l_70(t_46)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]
        t_51 = torch.add(t_11, t_47)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_52 = self.l_21(t_48)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_53 = self.l_37(t_49)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/Conv2d[conv]
        t_54 = self.l_71(t_50)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_55 = self.l_22(t_52)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_56 = self.l_38(t_53)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_57 = self.l_23(t_55)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_58 = self.l_39(t_56)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_59 = self.l_24(t_57)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_60 = self.l_40(t_58)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_61 = self.l_25(t_59)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_62 = self.l_41(t_60)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        t_63 = torch.add(x1, t_61)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        t_64 = torch.add(t_31, t_62)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[8]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/Sequential[layers]/MergeTwo[14]
        t_65 = [t_63, t_64, t_51]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/prim::ListConstruct4539
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/prim::Constant4540
        t_66 = torch.cat(t_65, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/aten::cat4541
        t_67 = self.l_53(t_66)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[4]/aten::cat4541
        t_68 = self.l_100(t_66)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5297
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5298
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5299
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5300
        t_69 = t_68[0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/ReLU[relu]
        t_70 = self.l_54(t_67)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/ReLU[relu]
        t_71 = self.l_101(t_68)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/aten::slice5301
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5302
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5303
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5304
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5305
        t_72 = t_69[:, 0:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/Conv2d[conv]
        t_73 = self.l_55(t_70)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/aten::slice5306
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5307
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5308
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5309
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5310
        t_74 = t_72[:, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_75 = self.l_56(t_73)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_76 = self.l_59(t_73)
        # calling Tensor.slice with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/aten::slice5311
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5312
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5313
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5314
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5315
        t_77 = t_74[:, :, :, 1:9223372036854775807:1]
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/MaxPool2d[pool]
        t_78 = self.l_57(t_75)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/AvgPool2d[pool]
        t_79 = self.l_60(t_76)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/aten::slice5316
        t_80 = self.l_102(t_77)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_81 = self.l_58(t_78)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_82 = self.l_61(t_79)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_1]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/Conv2d[conv_2]
        t_83 = [t_71, t_80]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::ListConstruct5336
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/prim::Constant5337
        t_84 = torch.cat(t_83, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[1]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_85 = torch.add(t_81, t_82)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/aten::cat5338
        t_86 = self.l_103(t_84)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[3]/Conv_3x3[module]/Conv_Cell[conv3_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]
        t_87 = torch.add(t_54, t_85)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[13]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_88 = torch.add(t_87, t_28)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]
        t_89 = self.l_73(t_85)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]
        t_90 = self.l_85(t_85)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[2]
        t_91 = self.l_89(t_85)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_92 = self.l_107(t_86)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_93 = self.l_110(t_86)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        t_94 = self.l_147(t_86)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[5]
        t_95 = self.l_92(t_87)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_96 = self.l_74(t_89)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/AvgPool2d[pool]
        t_97 = self.l_86(t_90)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/AvgPool2d[pool]
        t_98 = self.l_90(t_91)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/AvgPool2d[pool]
        t_99 = self.l_108(t_92)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/ReLU[relu]
        t_100 = self.l_111(t_93)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/AvgPool2d[pool]
        t_101 = self.l_148(t_94)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/ReLU[relu]
        t_102 = self.l_93(t_95)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_103 = self.l_75(t_96)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_104 = self.l_87(t_97)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_105 = self.l_91(t_98)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_106 = self.l_109(t_99)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/Conv2d[conv]
        t_107 = self.l_112(t_100)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_108 = self.l_149(t_101)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/Conv2d[conv]
        t_109 = self.l_94(t_102)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_110 = self.l_76(t_103)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[9]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[10]/Conv_Cell[module]/BatchNorm2d[norm]
        t_111 = torch.add(t_105, t_109)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[0]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[1]/Conv_Cell[module]/BatchNorm2d[norm]
        t_112 = torch.add(t_106, t_107)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_113 = self.l_77(t_110)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2]
        t_114 = self.l_114(t_112)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_115 = self.l_78(t_113)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/AvgPool2d[pool]
        t_116 = self.l_115(t_114)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_117 = self.l_79(t_115)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/Conv2d[conv]
        t_118 = self.l_116(t_116)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[2]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[4]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_119 = torch.add(t_112, t_118)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_120 = self.l_80(t_117)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[5]
        t_121 = self.l_150(t_119)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_122 = self.l_81(t_120)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/ReLU[relu]
        t_123 = self.l_151(t_121)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_124 = self.l_82(t_122)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/Conv2d[conv]
        t_125 = self.l_152(t_123)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[12]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[13]/Conv_Cell[module]/BatchNorm2d[norm]
        t_126 = torch.add(t_108, t_125)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_127 = self.l_83(t_124)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_128 = self.l_84(t_127)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[6]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/InputOne[7]/Pool_Operation[module]/Conv_Cell[conv_cell]/BatchNorm2d[norm]
        t_129 = torch.add(t_128, t_104)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[8]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/Sequential[layers]/MergeTwo[14]
        t_130 = [t_129, t_111, t_88]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/prim::ListConstruct5274
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/prim::Constant5275
        t_131 = torch.cat(t_130, 1)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[5]/aten::cat5276
        t_132 = self.l_104(t_131)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/ReLU[relu]
        t_133 = self.l_105(t_132)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/Conv2d[conv]
        t_134 = self.l_106(t_133)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_135 = self.l_118(t_134)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_136 = self.l_131(t_134)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Conv_Cell[preprocess1]/BatchNorm2d[norm]
        t_137 = self.l_134(t_134)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_138 = self.l_119(t_135)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/ReLU[relu]
        t_139 = self.l_132(t_136)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/ReLU[relu]
        t_140 = self.l_135(t_137)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_141 = self.l_120(t_138)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/Conv2d[conv]
        t_142 = self.l_133(t_139)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/Conv2d[conv]
        t_143 = self.l_136(t_140)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_144 = self.l_121(t_141)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv1_1x1]/BatchNorm2d[norm]
        t_145 = self.l_137(t_143)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_146 = self.l_122(t_144)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/ReLU[relu]
        t_147 = self.l_138(t_145)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_148 = self.l_123(t_146)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/Conv2d[conv]
        t_149 = self.l_139(t_147)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_150 = self.l_124(t_148)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv2_1x7]/BatchNorm2d[norm]
        t_151 = self.l_140(t_149)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_152 = self.l_125(t_150)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/ReLU[relu]
        t_153 = self.l_141(t_151)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_154 = self.l_126(t_152)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/Conv2d[conv]
        t_155 = self.l_142(t_153)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_156 = self.l_127(t_154)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv3_7x1]/BatchNorm2d[norm]
        t_157 = self.l_143(t_155)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_158 = self.l_128(t_156)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/ReLU[relu]
        t_159 = self.l_144(t_157)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_160 = self.l_129(t_158)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/Conv2d[conv]
        t_161 = self.l_145(t_159)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/FactorizedReduce[preprocess0]/BatchNorm2d[bn]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[7]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        t_162 = torch.add(t_86, t_160)
        # calling AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[9]/Conv_Cell[module]/BatchNorm2d[norm]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/InputOne[10]/Conv_7x1_1x7[module]/Conv_Cell[conv4_1x1]/BatchNorm2d[norm]
        t_163 = torch.add(t_142, t_161)
        # building a list from:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[8]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[11]
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/Sequential[layers]/MergeTwo[14]
        t_164 = [t_162, t_163, t_126]
        # calling torch.cat with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/prim::ListConstruct6069
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/prim::Constant6070
        t_165 = torch.cat(t_164, 1)
        # calling AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling] with arguments:
        # AmoebaNet_D/Sequential[cells]/Amoeba_Cell[6]/aten::cat6071
        t_166 = self.l_154(t_165)
        # calling Tensor.size with arguments:
        # AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling]
        # AmoebaNet_D/Classifier[classifier]/prim::Constant6085
        t_167 = Tensor.size(t_166, 0)
        # building a list from:
        # AmoebaNet_D/Classifier[classifier]/aten::size6086
        # AmoebaNet_D/Classifier[classifier]/prim::Constant6089
        t_168 = [t_167, -1]
        # calling Tensor.view with arguments:
        # AmoebaNet_D/Classifier[classifier]/AvgPool2d[global_pooling]
        # AmoebaNet_D/Classifier[classifier]/prim::ListConstruct6090
        t_169 = Tensor.view(t_166, t_168)
        # calling AmoebaNet_D/Classifier[classifier]/Linear[classifier] with arguments:
        # AmoebaNet_D/Classifier[classifier]/aten::view6091
        t_170 = self.l_155(t_169)
        # returing:
        # AmoebaNet_D/Classifier[classifier]/Linear[classifier]
        return (t_170,)

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
        result = dict()
        for k, v in state.items():
            if k in lookup:
                result[lookup[k]] = v
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                result[new_k] = v
        return result

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

