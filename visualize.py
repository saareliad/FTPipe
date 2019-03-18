import torch
import torchvision.models
import hiddenlayer as hl

model = torchvision.models.vgg16()

# Build HiddenLayer graph
# Jupyter Notebook renders it automatically
hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
