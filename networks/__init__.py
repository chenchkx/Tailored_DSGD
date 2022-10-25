
import torchvision
from .alexnet import AlexNet
from .densenet import DenseNet
from .resnet import *

import torch
torchvision.models.alexnet 
torchvision.models.densenet121 
def load_model(name, inputsize, outputsize):

    if name.lower() == 'resnet18':
        # resnet18 = torchvision.models.resnet18(pretrained=True) # resnet18 in torchvision 
        model = resnet18(num_classes=outputsize, pretrained=True) # resnet18 tailored for small input size
    if name.lower() == 'alexnet':
        model = AlexNet(num_classes=outputsize)
    if name.lower() == "densenet":
        model = DenseNet(num_classes=outputsize)

    return model



