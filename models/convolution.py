import torch.nn as nn
import torch.nn.functional as F

from utils import CONFIG


class StdConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def convolution(*args, **kwargs):
    if CONFIG["convolution"] == "std_conv":
        return StdConv(*args, **kwargs)
    elif CONFIG["convolution"] == "conv":
        return nn.Conv2d(*args, **kwargs)
    else:
        raise ValueError("Wrong value. (CONFIG)")
