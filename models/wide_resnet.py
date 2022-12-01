import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import CONFIG
from .activation import activation
from .convolution import convolution
from .normalization import normalization


def make_layer(block, in_planes, out_planes, num_blocks, stride):
    layers = []
    for idx in range(num_blocks):
        if idx == 0:
            layers.append(block(in_planes, out_planes, stride))
        else:
            layers.append(block(out_planes, out_planes, 1))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.conv1 = convolution(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.normalization1 = normalization(out_planes)
        self.conv2 = convolution(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.normalization2 = normalization(out_planes)
        self.activation = activation()

        self.shortcut = nn.Sequential()
        if (stride != 1) or (in_planes != out_planes):
            self.shortcut = nn.Sequential(
                convolution(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                normalization(out_planes)
            )

    def forward(self, x):
        out = self.activation(self.normalization1(self.conv1(x)))
        out = self.normalization2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class WideResNet(nn.Module):
    def __init__(self):
        super().__init__()
        if CONFIG["dataset"] == "cifar100":
            num_classes = 100
            self.register_buffer("mu", torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
            self.register_buffer("sigma", torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))
        elif CONFIG["dataset"] == "cifar10":
            num_classes = 10
            self.register_buffer("mu", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
            self.register_buffer("sigma", torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))
        else:
            raise ValueError("Wrong value. (CONFIG)")

        self.conv1 = convolution(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.normalization1 = normalization(16)
        self.activation = activation()

        self.layer1 = make_layer(BasicBlock,  16, 160, num_blocks=4, stride=1)
        self.layer2 = make_layer(BasicBlock, 160, 320, num_blocks=4, stride=2)
        self.layer3 = make_layer(BasicBlock, 320, 640, num_blocks=4, stride=2)
        self.linear = nn.Linear(640, num_classes)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = (x - self.mu) / self.sigma
        out = self.activation(self.normalization1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(len(out), -1)
        out = self.linear(out)
        return out
