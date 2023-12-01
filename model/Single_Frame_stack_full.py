import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
# Step 1: Define the Encoder using VGG19
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        # print(vgg16.features)
        # Remove the fully connected layers at the end
        self.features = vgg16.features
        
    def forward(self, x):
        return self.features(x)

# Step 2: Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Upsampling layers to reconstruct the image
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            # nn.tanh()
        )
        
    def forward(self, x):
        return self.decoder(x)

# Step 3: Combine Encoder and Decoder to create Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, frames):
        x = torch.cat(frames, dim=1)
        # x = frames
        x = self.conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return [x]

if __name__ == "__main__":
    model = Autoencoder().cuda()
    summary(model, input_size = (12,256,256)) 