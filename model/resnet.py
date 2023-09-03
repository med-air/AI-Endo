import torch
import torch.nn as nn
from torchvision import models, transforms
from utils.augment import EFDMix

class ResNet(torch.nn.Module):
    def __init__(self, out_channels=5, has_fc=True):
        super(ResNet, self).__init__()
        self.has_fc = has_fc
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        if has_fc:
            self.share.add_module("efdmix1", EFDMix(p=0.5, alpha=0.1))
        self.share.add_module("layer2", resnet.layer2)
        if has_fc:
            self.share.add_module("efdmix2", EFDMix(p=0.5, alpha=0.1))
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        if has_fc:
            self.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    # nn.Dropout(),
                                    nn.Linear(512, out_channels))
            self.embed = nn.Sequential(nn.Linear(2048, 128))

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        if self.has_fc:
            return self.fc(x), self.embed(x)
        else:
            return x
