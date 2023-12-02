import torch
import torch.nn as nn
import torchvision.models as models
from ConvGRU import ConvGRU
from torchsummary import summary

# Encoder using VGG Feature Extractor
class VGGEncoder(nn.Module):
    def __init__(self, original_model):
        super(VGGEncoder, self).__init__()
        # Remove the fully connected layers (classification head)
        self.features = nn.Sequential(*list(original_model.features.children())[:-1])

    def forward(self, x):
        return self.features(x)

# Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, input_size, bottleneck_size):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Linear(input_size, bottleneck_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.bottleneck(x)
        return x

# RNN for Temporal Information
class VideoRNN(nn.Module):
    def __init__(self, height, width, channels, hidden_dim, kernel_size, num_layers):
        super(VideoRNN, self).__init__()
        self.rnn = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=torch.float,
                    batch_first=True,
                    bias = True,
                    return_all_layers = True) 
        # self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, cn) = self.rnn(x)
        # output = self.fc(hn[-1])
        return output

# Decoder to Regenerate Image
class Decoder(nn.Module):
    def __init__(self, input_size, output_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_size, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

# Full Pipeline combining the above classes
class VideoAnalyticsPipeline(nn.Module):
    def __init__(self, encoder, bottleneck, rnn, decoder):
        super(VideoAnalyticsPipeline, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.rnn = rnn
        self.decoder = decoder

    def forward(self, x):
        # Spatial Encoding
        x = self.encoder(x)
        
        # Temporal Bottleneck
        x = self.bottleneck(x)

        # RNN for Temporal Information
        # x = x.unsqueeze(1)  # Add temporal dimension
        x = self.rnn(x)

        # Decoder to Regenerate Image
        x = self.decoder(x)
        return x
if __name__ == "__main__":
    # Set up the full pipeline with VGG16
    vgg_encoder = models.vgg16(pretrained=True)
    encoder = VGGEncoder(vgg_encoder.features)
    bottleneck = Bottleneck(512, 256)  # Adjust the bottleneck size as needed
    rnn = VideoRNN(height=7, width=7, channels=512, hidden_dim=[32,64], kernel_size=(3,3), num_layers=2)  # Adjust based on your requirements
    decoder = Decoder(512, 3)  # 128 is the size of the bottleneck, 3 is the number of channels in the output image
    video_pipeline = VideoAnalyticsPipeline(encoder, bottleneck, rnn, decoder)
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_pipeline = video_pipeline.to(device)
    summary(video_pipeline, input_size =(3,256,256))

# Example usage:
# Assuming you have a video sequence tensor 'video_sequence' (batch_size, sequence_length, channels, height, width)
# output = video_pipeline(video_sequence)
