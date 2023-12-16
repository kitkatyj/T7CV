import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
try:
    from .pretrained.FlowNetS import FlowNetS
except:
    from pretrained.FlowNetS import FlowNetS

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
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # frames (1,3*,256,256) concat along dim 1 
        # x = torch.cat(frames, dim=1) # --> (1,12,256,256)
        x = self.conv(x)
        skip1, skip2, skip3, skip4 = self.encoder(x)
        # warped_skip1 = self.warp(skip1,flow2)
        # warped_skip2 = self.warp(skip2,flow3)
        # warped_skip3 = self.warp(skip3,flow4)
        # warped_skip4 = self.warp(skip4,flow5)
        x = self.decoder(skip1, skip2, skip3, skip4)
        return x
    
    def warp(self,feature_map, optical_flow):
        """
        Warp a feature map using an optical flow map.

        Args:
        - feature_map (torch.Tensor): Input feature map of shape (batch_size, channels, height, width).
        - optical_flow (torch.Tensor): Optical flow map of shape (batch_size, 2, height, width).

        Returns:
        - warped_feature_map (torch.Tensor): Warped feature map of the same shape as the input feature map.
        """
        batch_size, _, height, width = feature_map.shape

        # Generate a grid of coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        grid_y = grid_y.float().cuda()
        grid_x = grid_x.float().cuda()

        # Normalize coordinates to [-1, 1]
        normalized_grid_y = (grid_y - (height - 1) / 2) / ((height - 1) / 2)
        normalized_grid_x = (grid_x - (width - 1) / 2) / ((width - 1) / 2)

        # Stack the normalized coordinates
        normalized_grid = torch.stack([normalized_grid_x, normalized_grid_y], dim=-1)

        # Add optical flow to the normalized coordinates
        warped_normalized_grid = normalized_grid + optical_flow.permute(0, 2, 3, 1)

        # Normalize back to pixel coordinates
        warped_grid_x = warped_normalized_grid[:, :, :, 0] * ((width - 1) / 2) + (width - 1) / 2
        warped_grid_y = warped_normalized_grid[:, :, :, 1] * ((height - 1) / 2) + (height - 1) / 2

        # Stack the warped pixel coordinates
        warped_grid = torch.stack([warped_grid_x, warped_grid_y], dim=-1)

        # Use grid_sample to warp the feature map
        warped_feature_map = F.grid_sample(feature_map, warped_grid, align_corners=False)

        return warped_feature_map


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
        self.unet = UNet()

        self.interpolate_frames = InterpolateFrames()

    def forward(self, frames, t=0.5):
        # Stack four frames along the channel dimension
        x = torch.cat(frames, dim=1)
        # x = frames

        flow2,flow3,flow4,flow5,flow6 = self.flownet(x)
        # flow2 = self.upsample1(flow2)
        # flow3 = self.upsample1(flow3)
        # flow4 = self.upsample1(flow4)
        # flow5 = self.upsample1(flow5)
        # flow6 = self.upsample1(flow6)
        
        # Interpolate optical flows to get intermediate flow at time t
        interp_flow = self.interpolate_frames(flow2, flow3, flow4, flow5, flow6, t)

        interp_flow = self.upsample1(interp_flow)

        # Warp frame2 and frame3 using the interpolated flow
        warped_frame2 = self.warp(frames[1], interp_flow)
        warped_frame3 = self.warp(frames[2], interp_flow)

        # Combine warped frames based on interpolation parameter t
        interpolated_frame = torch.cat([warped_frame2, warped_frame3], dim=1)
        interpolated_frame = self.unet(interpolated_frame) #, flow2, flow3, flow4, flow5)
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
    inputs = [torch.randn((1, 3, 256, 256)).cuda() for _ in range(4)]
    # summary(model, input_size = (12, 256, 256), device  = 'cuda')
    x = model(inputs)[0]
    print(x.size())
    num_params = sum(p.numel() for p in model.parameters()) 
    print("Number of parameters in the model:", num_params)
