
import torchvision.models
from .alexnet import *
from .densenet import *
from .resnet import *
from .vgg import *



def load_model(name, inputsize, outputsize):

    if name.lower() == 'resnet18':
        # resnet18 = torchvision.models.resnet18(pretrained=True) # resnet18 in torchvision 
        model = resnet18(num_classes=outputsize, pretrained=True) # resnet18 tailored for small input size
    if name.lower() == 'alexnet':
        model = alexnet(num_classes=outputsize, pretrained=True)
    if name.lower() == "densenet121":
        model = densenet121(num_classes=outputsize, pretrained=True)
    if name.lower() == 'vgg11_bn':
        model = vgg11_bn(num_classes=outputsize, pretrained=False)
    if name.lower() == 'vgg11':
        model = vgg11(num_classes=outputsize, pretrained=True)
    if name.lower() == 'vgg13_bn':
        model = vgg13_bn(num_classes=outputsize, pretrained=True)


    return model



