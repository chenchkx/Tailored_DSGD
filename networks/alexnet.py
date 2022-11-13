import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Any


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=3), # kernel_size: 11->7, stride: 4->1, padding: 2->3
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # kernel_size: 11->7, stride: 4->1, padding: 2->3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        import copy
        model_dict = copy.deepcopy(model.state_dict())
        for name, param in state_dict.items():
            if name in ['features.0.weight','features.0.bias']:
                model_dict[name] = param + torch.std(param)*torch.rand_like(param)
            #     continue
            if name in ['features.3.weight']:
                model_dict['features.4.weight'] =  param
            elif name in ['features.6.weight']:
                model_dict['features.7.weight'] =  param
            elif name in ['features.8.weight']:
                model_dict['features.9.weight'] =  param
            elif name in ['features.10.weight']:
                model_dict['features.12.weight'] =  param
            elif name in ['features.3.bias']:
                model_dict['features.4.bias'] =  param
            elif name in ['features.6.bias']:
                model_dict['features.7.bias'] =  param
            elif name in ['features.8.bias']:
                model_dict['features.9.bias'] =  param
            elif name in ['features.10.bias']:
                model_dict['features.12.bias'] =  param
            elif 'classifier' in name:
                continue
            else:
                model_dict[name] =  param
        model.load_state_dict(model_dict)
    return model