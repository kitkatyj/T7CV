import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_3D import SEGating

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=False):
        super().__init__()

        self.branch1x1 = Conv_3d(in_channels, out_channels, kernel_size=1, bias=True, batchnorm=batchnorm)

        self.branch3x3 = nn.Sequential(
            Conv_3d(in_channels, out_channels, kernel_size=1, bias=True, batchnorm=batchnorm),
            Conv_3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True, batchnorm=batchnorm)
        )

        self.branch5x5 = nn.Sequential(
            Conv_3d(in_channels, out_channels, kernel_size=1, bias=True, batchnorm=batchnorm),
            Conv_3d(out_channels, out_channels, kernel_size=5, padding=2, bias=True, batchnorm=batchnorm)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            Conv_3d(in_channels, out_channels, kernel_size=1, bias=True, batchnorm=batchnorm)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        print(branch1x1.shape, 'br1')
        branch3x3 = self.branch3x3(x)
        print(branch3x3.shape, 'branch3x3')
        branch5x5 = self.branch5x5(x)
        print(branch5x5.shape, 'branch5x5')
        branch_pool = self.branch_pool(x)
        print(branch_pool.shape, 'branch_pool')
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)