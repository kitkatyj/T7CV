import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .pretrained.FlowNetS import FlowNetS

from torchsummary import summary

# Step 1: Define the Encoder using VGG16
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        
        # Remove the fully connected layers at the end
        self.features = vgg16.features[:-1]
        
    def forward(self, x):
        return self.features(x)

# Step 2: Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Upsampling layers to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.decoder(x)


# Step 3: Combine Encoder and Decoder to create Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        # x = torch.cat(frames, dim=1)
        # x = frames
        x = self.conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FlowNetS_Interpolation(nn.Module):
    def __init__(self, batchNorm=True, pretrained = None):
        super(FlowNetS_Interpolation, self).__init__()

        self.flownet = FlowNetS(input_channels=12, batchNorm=True)
        if pretrained:
            # Load the checkpoint
            checkpoint = torch.load(pretrained)

            # Manually adjust the size of conv1.0.weight


            model_state_dict = self.flownet.state_dict()

            # Filter out unnecessary keys from the checkpoint
            filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict and k != "conv1.0.weight"}

            # Load the filtered state_dict into the model
            self.flownet.load_state_dict(filtered_state_dict, strict=False)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.AE = Autoencoder()

        self.interpolate_frames = InterpolateFrames()

    def forward(self, frames, t=0.5):
        # Stack four frames along the channel dimension
        x = torch.cat(frames, dim=1)
        # x = frames

        flow2,flow3,flow4,flow5,flow6 = self.flownet(x)

        # Interpolate optical flows to get intermediate flow at time t
        interp_flow = self.interpolate_frames(flow2, flow3, flow4, flow5, flow6, t)

        interp_flow = self.upsample1(interp_flow)

        # Warp frame2 and frame3 using the interpolated flow
        warped_frame2 = self.warp(frames[1], interp_flow)
        warped_frame3 = self.warp(frames[2], interp_flow)

        # Combine warped frames based on interpolation parameter t
        interpolated_frame = torch.cat([warped_frame2, warped_frame3], dim=1)
        interpolated_frame = self.AE(interpolated_frame)
        return [interpolated_frame]

    def warp(self, x, flow):
        _, _, w, h = flow.size()
        grid_x, grid_y = torch.meshgrid(torch.arange(0, w), torch.arange(0, h)) # (448, 256)
        grid_x = grid_x.float().to(x.device)
        grid_y = grid_y.float().to(x.device)

        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        warped_grid_x = grid_x.unsqueeze(0).expand_as(u) + u
        warped_grid_y = grid_y.unsqueeze(0).expand_as(v) + v

        # Normalize coordinates to be in the range [-1, 1]
        normalized_grid_x = (2.0 * warped_grid_x / (w - 1)) - 1.0
        normalized_grid_y = (2.0 * warped_grid_y / (h - 1)) - 1.0

        # Stack coordinates along the channel dimension
        grid = torch.stack((normalized_grid_x, normalized_grid_y), dim=3)

        # Sample pixels from the input image using the warped coordinates
        # Use 'zeros' padding_mode to handle out-of-bound values with zeros
        warped_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_x


class InterpolateFrames(nn.Module):
    def __init__(self):
        super(InterpolateFrames, self).__init__()

    def forward(self, flow2, flow3, flow4, flow5, flow6, t):
        # Interpolate optical flows at different scales
        interp_flow5 = t * flow5 + (1 - t) * F.interpolate(flow6, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow4 = t * flow4 + (1 - t) * F.interpolate(interp_flow5, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow3 = t * flow3 + (1 - t) * F.interpolate(interp_flow4, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow2 = t * flow2 + (1 - t) * F.interpolate(interp_flow3, scale_factor=2, mode='bilinear', align_corners=False)
        return interp_flow2
    
if __name__ == "__main__":
    model = FlowNetS_Interpolation().cuda()
    # inputs = [torch.randn((1, 3, 256, 256)).cuda() for _ in range(4)]
    # # summary(model, input_size = (12, 256, 256), device  = 'cuda')
    # x = model(inputs)[0]
    # print(x.size())
