import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
try:
    from ConvLSTM import ConvLSTM
except:
    from .ConvLSTM import ConvLSTM

from torchsummary import summary
# Step 1: Define the Encoder using VGG16
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        
        self.features = vgg16.features
        
    def forward(self, x):
        return self.features(x)

class Encoder_nopool(nn.Module):
    def __init__(self):
        super(Encoder_nopool, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        
        self.features = vgg16.features[:-1]
        
    def forward(self, x):
        return self.features(x)
    
# Step 2: Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Upsampling layers to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
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
    

class Decoder_nopool(nn.Module):
    def __init__(self):
        super(Decoder_nopool, self).__init__()
        
        # Upsampling layers to reconstruct the image
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(inplace=True),
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

class Temporal_bottleneck(nn.Module):
    def __init__(self):
        super(Temporal_bottleneck, self).__init__()
        self.rnn = ConvLSTM(input_dim=512,
                    hidden_dim=[32,64, 128, 256],
                    kernel_size=(3,3),
                    num_layers=4,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)
        self.rnn2 = ConvLSTM(input_dim=512,
                    hidden_dim=[32,64, 128, 256],
                    kernel_size=(3,3),
                    num_layers=4,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)
    def forward(self, x):
        lr = self.rnn(x)[0][0][:,2,:,:,:]
        x = torch.flip(x, dims=[1])
        rl = self.rnn2(x)[0][0][:,2,:,:,:]
        x = torch.cat([lr,rl], dim=1)
        return x
    
class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding = 1)
        self.conv1d2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding = 1)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1d1(x))
        x = F.relu(self.conv1d2(x))
        x = x.permute(0,2,1)
        return x

# Step 3: Combine Encoder and Decoder to create Autoencoder
class MIM_LSTM(nn.Module):
    def __init__(self):
        super(MIM_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.encoder = Encoder()
        self.temporal = Temporal_bottleneck()
        self.decoder = Decoder()
        
    def forward(self, frames:list):
        frame25 = (frames[1] +frames[0])/2
        frames.insert(2,frame25)
        x = torch.stack(frames, dim=1)
        # x = self.conv(x)
        # x = frames
        batch_size, num_frames, channels, height, width = x.size()
        # Reshape to (batch_size * num_frames, channels, height, width)
        x = x.view(-1, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, num_frames, *x.shape[1:])
        x = self.temporal(x)
        x = self.decoder(x)
        return [x]

class MIM_transformer(nn.Module):
    def __init__(self):
        super(MIM_transformer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.encoder = Encoder_nopool()
        self.bottleneck1 = nn.Linear(512*8*8, 256)
        self.pool = torch.nn.AdaptiveAvgPool2d((8,8))
        self.bottleneck2 = nn.Linear(256, 512*8*8)
        # self.temporal = Temporal_bottleneck()
        self.temporal = nn.MultiheadAttention(embed_dim = 256, num_heads = 8, batch_first = True)
        self.decoder = Decoder_nopool()
        
    def forward(self, frames:list):
        frame25 = (frames[1] +frames[2])/2
        frames.insert(2,frame25)
        x = torch.stack(frames, dim=1)
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        # x =self.encoder(x)
        # _, feat_channels, feat_height, feat_width = x.size()
        # x = self.pool(x).view(x.size(0), -1) ##
        features =self.encoder(x)
        _, feat_channels, feat_height, feat_width = features.size()
        x = self.pool(features).view(features.size(0), -1)
        x = F.relu(self.bottleneck1(x))
        x = x.view(batch_size, num_frames, *x.shape[1:])
        x, _ = self.temporal(x, x,x) 
        x = x[:, 2, :].unsqueeze(1)
        x = x.view(-1, 256)
        x = F.relu(self.bottleneck2(x))
        x = x.view(-1, 8,8)
        x = F.interpolate(x.unsqueeze(1), size=(feat_height,feat_width), mode='bilinear').squeeze(1)
        x = x.view(batch_size, feat_channels, feat_height, feat_width)
        features = features.view(batch_size,-1,feat_channels, feat_height, feat_width)[:,2,:,:,:].squeeze(1)
        x = x+ features
        x = self.decoder(x)
        return [x]
    
class MIM_TCN(nn.Module):
    def __init__(self):
        super(MIM_TCN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.encoder = Encoder_nopool()
        self.bottleneck1 = nn.Linear(512*8*8, 256)
        self.pool = torch.nn.AdaptiveAvgPool2d((8,8))
        self.bottleneck2 = nn.Linear(256, 512*8*8)
        # self.temporal = Temporal_bottleneck()
        # self.temporal = nn.MultiheadAttention(embed_dim = 512, num_heads = 8, batch_first = True)
        self.temporal = TCN()
        self.decoder = Decoder_nopool()
        
    def forward(self, frames:list):
        frame25 = (frames[1] +frames[2])/2
        frames.insert(2,frame25)
        x = torch.stack(frames, dim=1)
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        features =self.encoder(x)
        _, feat_channels, feat_height, feat_width = features.size()
        x = self.pool(features).view(features.size(0), -1)
        x = F.relu(self.bottleneck1(x))
        x = x.view(batch_size, num_frames, *x.shape[1:])
        x = self.temporal(x) 
        x = x[:, 2, :].unsqueeze(1)
        x = x.view(-1, 256)
        x = F.relu(self.bottleneck2(x))
        x = x.view(-1, 8,8)
        x = F.interpolate(x.unsqueeze(1), size=(feat_height,feat_width), mode='bilinear').squeeze(1)
        x = x.view(batch_size, feat_channels, feat_height, feat_width)
        features = features.view(batch_size,-1,feat_channels, feat_height, feat_width)[:,2,:,:,:].squeeze(1)
        x = x+ features
        x = self.decoder(x)
        return [x]

if __name__ == "__main__":
    # model = Temporal_bottleneck_1dCNN().cuda()
    # model = MIM_LSTM()
    # model = Conv_SelfAttention(512).cuda()
    # random_tensor = torch.randn(2, 5, 128, 4, 4).cuda()
    # # x = model([random_tensor,random_tensor,random_tensor,random_tensor])
    # x = model(random_tensor)
    # print(x.shape)
    model = MIM_transformer()
    x = model([torch.randn((2, 3, 256, 256)) for _ in range(4)])
    # x = model(x)
    print(x[0].shape)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

        # Example: Count parameters in your transformer_model
    num_params = count_parameters(model)
    print(num_params)
    # summary(model, input_size = (4,3,256,256))# (512,16,16)) 