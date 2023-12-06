import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

# Step 1: Define the Encoder using VGG16
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        features = list(models.vgg16(pretrained = True).features)[:23]
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22}:
                results.append(x)
        return results

# Step 2: Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Upsampling layers to reconstruct the image
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,  skip1, skip2, skip3, skip4):
        # x = F.relu(self.upconv4(x))  # Upsample
        # x = torch.cat((x, skip4), dim=1)  # Concatenate with skip connection
        # x = F.relu(self.conv4(x))
        x = skip4

        x = F.relu(self.upconv3(x))  # Upsample
        x = torch.cat((x, skip3), dim=1)  # Concatenate with skip connection
        x = F.relu(self.conv3(x))

        x = F.relu(self.upconv2(x))  # Upsample
        x = torch.cat((x, skip2), dim=1)  # Concatenate with skip connection
        x = F.relu(self.conv2(x))

        x = F.relu(self.upconv1(x))  # Upsample
        x = torch.cat((x, skip1), dim=1)  # Concatenate with skip connection
        x = F.relu(self.conv1(x))

        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x


# Step 3: Combine Encoder and Decoder to create U-Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, frames):
        # frames (1,3*,256,256) concat along dim 1 
        x = torch.cat(frames, dim=1) # --> (1,12,256,256)
        x = self.conv(x)
        skip1, skip2, skip3, skip4 = self.encoder(x)
        x = self.decoder(skip1, skip2, skip3, skip4)
        return [x]

if __name__ == "__main__":
    model = UNet()
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters in the model:", num_params)
    # summary(model, input_size = (12,256,256)) 
    # inputs = [torch.randn((2, 3,256,256)) for _ in range(4)]
    # x = model(inputs)
    # print(x[0].shape)