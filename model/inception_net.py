import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['unet_18', 'unet_34']

useBias = False

class identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x

class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=useBias)

    @staticmethod
    def get_downsample_stride(stride, temporal_stride):
        if temporal_stride:
            return (temporal_stride, stride, stride)
        else:
            return (stride, stride, stride)

class BasicStem(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                padding=(1, 3, 3), bias=useBias),
            nn.BatchNorm3d(64),  # Assuming batchnorm is nn.BatchNorm3d
            nn.ReLU(inplace=False))

class Inception3d(nn.Module):
    def __init__(self, in_channels):
        super(Inception3d, self).__init__()
        self.branch1x1 = nn.Conv3d(in_channels, 16, kernel_size=1)
        self.branch3x3 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv3d(in_channels, 8, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv3d(in_channels, 8, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        if not isinstance(stride , int):
            temporal_stride , stride , stride = stride
        else:
            temporal_stride = stride

        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            # batchnorm(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(temporal_stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride , temporal_stride):
        if temporal_stride:
            return (temporal_stride, stride, stride)
        else:
            return (stride , stride , stride)

class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super().__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            batchnorm(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            batchnorm(64),
            nn.ReLU(inplace=True))


class SEGating(nn.Module):

    def __init__(self , inplanes , reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes , inplanes , kernel_size=1 , stride=1 , bias=True),
            nn.Sigmoid()
        )
        
    def forward(self , x):

        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            batchnorm(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            batchnorm(planes)
        )
        self.fg = SEGating(planes) ## Feature Gating
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class VideoResNet(nn.Module):
    def __init__(self, block, conv_makers, layers, stem, zero_init_residual=False):
        super(VideoResNet, self).__init__()
        self.inplanes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.inception = Inception3d(64)  # Use 64, as this is the output of layer1
        # self.inplanes = 256  # Adjusting inplanes after Inception
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[0], stride=2, temporal_stride=1)
        self.layer3 = self._make_layer(block, conv_makers[1], 256, layers[1], stride=2, temporal_stride=1)
        self.layer4 = self._make_layer(block, conv_makers[2], 512, layers[2], stride=1, temporal_stride=1)
        # self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=1, temporal_stride=2)
        self._initialize_weights()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 0)

    def forward(self, x):
        x_0 = self.stem(x)
        x_1 = self.inception(x_0)  # Applying Inception module
        x_1 = self.layer1(x_1)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        # Not returning the last layer x_5
        return x_0, x_1, x_2, x_3, x_4

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, temporal_stride=None):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride , temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                batchnorm(planes * block.expansion)
            )
            stride = ds_stride

        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample ))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)
    # TODO: Implement pretrained model loading
    return model

def unet_18(pretrained=False, bn=False, progress=True, **kwargs):
    """
    Construct 18 layer Unet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 encoder
    """
    global batchnorm
    if bn:
        batchnorm = nn.BatchNorm3d
    else:
        batchnorm = identity

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)

def unet_34(pretrained=False, bn=False, progress=True, **kwargs):
    """
    Construct 34 layer Unet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 encoder
    """
    global batchnorm
    # bn = False
    if bn:
        batchnorm = nn.BatchNorm3d
    else:
        batchnorm = identity


    return _video_resnet('r3d_34',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[3, 4, 6, 3],
                         stem=BasicStem, **kwargs)
