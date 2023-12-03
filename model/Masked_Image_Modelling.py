import torch
import torch.nn as nn
import torchvision.models as models
from ConvLSTM import ConvLSTM

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
        super(Encoder, self).__init__()
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
    
class Temporal_bottleneck_1dCNN(nn.Module):
    def __init__(self):
        super(Temporal_bottleneck_1dCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=128*4*4, out_channels=128*4*4, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # x = x.permute(0, 2, 1, 3, 4).contiguous()  # Reshape for 1D convolution
        print(x.shape)
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size, num_frames, -1)
        x = x.permute(0,2,1).contiguous()
        print(x.shape)
        x = self.conv1d(x)
        print(x.shape)
        x = x.view(batch_size, num_frames, -1, height, width)
        # x = x.permute(0, 2, 1, 3, 4).contiguous()  # Reshape back to the original shape
        print(x.shape)
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
        super(MIM_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.encoder = Encoder_nopool()
        self.pool = torch.nn.AdaptiveAvgPool2d((8,8))
        # self.temporal = Temporal_bottleneck()
        self.temporal = nn.MultiheadAttention(embed_dim = 512*8*8)
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

if __name__ == "__main__":
    model = Temporal_bottleneck_1dCNN().cuda()
    # model = Conv_SelfAttention(512).cuda()
    random_tensor = torch.randn(2, 5, 128, 4, 4).cuda()
    # x = model([random_tensor,random_tensor,random_tensor,random_tensor])
    x = model(random_tensor)
    print(x.shape)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Example: Count parameters in your transformer_model
    num_params = count_parameters(model)
    print(num_params)
    # summary(model, input_size = (4,3,256,256))# (512,16,16)) 