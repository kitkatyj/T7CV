import torch
import torch.nn as nn
from torchvision.models import resnet18, densenet121, alexnet
import torch.nn.functional as F

class SimpleResNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
class SimpleDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleDenseNet, self).__init__()
        self.densenet = densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)
#vgg; 
class SimpleAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleAlexNet, self).__init__()
        self.alexnet = alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return F.sigmoid(out) * x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.ca(out)
        out = self.conv2(out)
        out += residual
        return out

class SimpleCAIN(nn.Module):
    def __init__(self, in_channels, num_blocks=5):
        super(SimpleCAIN, self).__init__()
        self.entry_conv = nn.Conv2d(in_channels, 64, kernel_size=7, padding=3)

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])

        self.exit_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.entry_conv(x))
        out = self.res_blocks(out)
        out = self.exit_conv(out)
        return out
    
# Encoder-Decoder Hybrid
   # The encoder can capture the necessary features from the input frames, 
   # while the decoder can focus on reconstructing the intermediate frame. 
   # By appropriately choosing the encoder and decoder components from 
   # ResNet, DenseNet, and AlexNet, this hybrid can be very effective.
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Using AlexNet as the initial part of the encoder
        # Purpose of AlexNet: AlexNet is used here for its ability to capture basic features from the input frames. It acts as the first stage in the feature extraction process.
        self.alexnet_features = alexnet(pretrained=True).features

        # Using ResNet18 as the second part of the encoder
        # This uses the ResNet18 architecture to extract features from the input frames. The fully connected layers are removed, as we are only interested in the feature maps.
        resnet = resnet18(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[4:-2])
        self.avgpool = resnet.avgpool

    def forward(self, x):
        x = self.alexnet_features(x)
        x = self.resnet_layers(x)
        x = self.avgpool(x)
        return x

class Decoder(nn.Module):

    def __init__(self, num_classes=1000):
        super(Decoder, self).__init__()
        # Using DenseNet121 as a decoder
        # This part uses DenseNet121 to reconstruct the output from the features provided by the encoder. It includes the DenseNet's classification layers, but you might need to modify this depending on your specific frame interpolation task.
        densenet = densenet121(pretrained=True)
        self.features = densenet.features
        self.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EncoderDecoderHybrid(nn.Module):
    def __init__(self, num_classes=1000):
        super(EncoderDecoderHybrid, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
