import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)