import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, num_channels, num_heads, hidden_size=None):
        super(MultiHeadAttentionModel, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_heads,
            kdim=None,  # Set to None to use input dimension
            vdim=None,  # Set to None to use input dimension
        )

        # self.linear1 = nn.Linear(num_channels, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, num_channels)

    def forward(self, x):
        # Assuming x has dimensions (batch_size, num_channels, height, width)
        batch_size, num_channels, height, width = x.shape
        # Reshape to (batch_size * height * width, num_channels)
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x, x, x)

        # Reshape back to original shape
        attn_output = attn_output.permute(1, 2, 0)
        x = attn_output.reshape(batch_size, num_channels, height, width)
        return x

    # def forward(self, x):
    #     # Assuming x has dimensions (batch_size, num_channels, height, width)
    #     batch_size, num_channels, height, width = x.shape
    #     # Reshape to (batch_size * height * width, num_channels)
    #     x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)
        
    #     # Apply multi-head attention
    #     attn_output, _ = self.multihead_attn(x, x, x)

    #     # Reshape back to original shape
    #     attn_output = attn_output.permute(1, 2, 0)
    #     x = attn_output.reshape(batch_size, num_channels, height, width)
    #     return x
    
    def forward(self, x):
        # Assuming x has dimensions (batch_size, num_channels, height, width)
        # batch_size, num_channels, height, width = x.shape
        # Reshape to (batch_size * height * width, num_channels)
        x2 = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x2, x2, x2)

        # Reshape back to original shape
        attn_output = attn_output.permute(1, 2, 0)
        x = attn_output.reshape(*x.shape)
        return x


def conv_with_attention(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, num_heads=2):
    if batchNorm:
        return nn.Sequential(
            MultiHeadAttentionModel(num_channels=in_planes, num_heads=num_heads),
            nn.ZeroPad2d((kernel_size - 1) // 2),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1),
        )
    else:
        return nn.Sequential(
            MultiHeadAttentionModel(num_channels=in_planes, num_heads=num_heads),
            nn.ZeroPad2d((kernel_size - 1) // 2),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.LeakyReLU(0.1),
        )

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.ZeroPad2d((kernel_size-1)//2),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.ZeroPad2d((kernel_size-1)//2),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=0, bias=True),
            nn.LeakyReLU(0.1)
        )

def deconv_with_attention(in_planes, out_planes, num_heads=2):
    return nn.Sequential(
        MultiHeadAttentionModel(num_channels=in_planes, num_heads=num_heads),
        nn.ZeroPad2d(0),
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1),
    )

def predict_flow(in_planes):
    return nn.Sequential(
        nn.ZeroPad2d(1),
        nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=0, bias=False)
    )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ZeroPad2d(0),
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1)
    )

class FlowNetS_att_Interpolation(nn.Module):
    def __init__(self, batchNorm=True):
        super(FlowNetS_att_Interpolation, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 12, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv_with_attention(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv_with_attention(self.batchNorm, 512, 512)
        self.conv5 = conv_with_attention(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv_with_attention(self.batchNorm, 512, 512)
        self.conv6 = conv_with_attention(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv_with_attention(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv_with_attention(1024, 512)
        self.deconv4 = deconv_with_attention(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.deconv1 = deconv(194, 12)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(78)


        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)

        self.interpolate_frames = InterpolateFrames()

    def forward(self, frames, t=0.5):
        # Stack four frames along the channel dimension
        x = torch.cat(frames, dim=1)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = F.interpolate(flow6, scale_factor=2, mode='bilinear', align_corners=False)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=False)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=False)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=False)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        flow2_up = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=False)
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(concat1)

        # Interpolate optical flows to get intermediate flow at time t
        interp_flow = self.interpolate_frames(flow1, flow2, flow3, flow4, flow5, flow6, t)


        # Warp frame2 and frame3 using the interpolated flow
        warped_frame2 = self.warp(frames[1], interp_flow)
        warped_frame3 = self.warp(frames[2], interp_flow)

        # Combine warped frames based on interpolation parameter t
        interpolated_frame = (1 - t) * warped_frame2 + t * warped_frame3
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

    def forward(self, flow1, flow2, flow3, flow4, flow5, flow6, t):
        # Interpolate optical flows at different scales
        interp_flow5 = t * flow5 + (1 - t) * F.interpolate(flow6, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow4 = t * flow4 + (1 - t) * F.interpolate(interp_flow5, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow3 = t * flow3 + (1 - t) * F.interpolate(interp_flow4, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow2 = t * flow2 + (1 - t) * F.interpolate(interp_flow3, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow1 = t * flow1 + (1 - t) * F.interpolate(interp_flow2, scale_factor=2, mode='bilinear', align_corners=False)
        interp_flow = F.interpolate(interp_flow1, scale_factor=2, mode='bilinear', align_corners=False)
        return interp_flow
    
if __name__ == "__main__":
    model = FlowNetS_att_Interpolation()
    num_params = sum(p.numel() for p in model.parameters()) 
    print("Number of parameters in the model:", num_params)