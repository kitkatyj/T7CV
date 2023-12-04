import torch
import torch.nn as nn
import torchvision.models as models
from .ConvGRU import ConvGRU
from torchsummary import summary
import numpy as np

# Encoder using VGG Feature Extractor
class VGGEncoder(nn.Module):
    def __init__(self, device, original_model):
        super(VGGEncoder, self).__init__()
        # Remove the fully connected layers (classification head)
        self.features = nn.Sequential(*list(original_model.features.children())[:-1])
        self.cuda()

    def forward(self, x):
        return self.features(x)

# Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, device, input_size, bottleneck_size):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Linear(input_size, bottleneck_size)
        self.bottleneck.cuda()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.bottleneck(x)
        return x

# RNN for Temporal Information
class VideoRNN(nn.Module):
    def __init__(self, device, height, width, channels, hidden_dim, kernel_size, num_layers):
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
        self.rnn.cuda()

    def forward(self, x):
        # _, (hn, cn) = self.rnn(x)
        # output = self.fc(hn[-1])
        layer_output_list, last_state_list = self.rnn(x)
        return last_state_list[-1][-1]


# Decoder to Regenerate Image
class Decoder(nn.Module):
    def __init__(self, device, input_size, output_channels):
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
        self.decoder.cuda()

    def forward(self, x):
        return self.decoder(x)

# Full Pipeline combining the above classes
class VideoAnalyticsPipeline(nn.Module):
    def __init__(self, device, encoder, bottleneck, rnn, decoder):
        super(VideoAnalyticsPipeline, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.rnn = rnn
        self.decoder = decoder
        self.cuda()

    def forward(self, x):
        # Assuming x is a list of tensors
        device = next(self.parameters()).device  # Get the device from the model's parameters
        # Move each tensor in the input list to the device
        x = [img_.cuda() for img_ in x]
        ############################ PT 1 - START ############################
        #TENSOR LIST
        # list_length = len(x)
        # print("Number of elements in the list:", list_length)
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        # # Number of elements in the list: 4
        # Shape of tensor 0: torch.Size([1, 3, 256, 256])
        # Shape of tensor 1: torch.Size([1, 3, 256, 256])
        # Shape of tensor 2: torch.Size([1, 3, 256, 256])
        # Shape of tensor 3: torch.Size([1, 3, 256, 256])
        
        #NPY
        # print(x.shape) # torch.Size([2, 512, 16, 16])
        ############################ PT 1 - END ############################

        ############################ PT 2 - START ############################
        #TENSOR LIST
        # Concatenate the tensors along the batch dimension
        x = torch.cat(x, dim=0)  # x is a list of tensors
        ############################ PT 2 - END ############################

        ############################ PT 3 - START ############################
        # Spatial Encoding
        x = self.encoder(x)

        #TENSOR LIST
        list_length = len(x)
        print("Number of elements in the list:", list_length)
        for i, tensor in enumerate(x):
            print(f"Shape of tensor {i}: {tensor.shape}")
        # Number of elements in the list: 4
        # Shape of tensor 0: torch.Size([512, 16, 16])
        # Shape of tensor 1: torch.Size([512, 16, 16])
        # Shape of tensor 2: torch.Size([512, 16, 16])
        # Shape of tensor 3: torch.Size([512, 16, 16])

        #array
        # print(x.shape) # torch.Size([2, 512, 16, 16])
        ############################ PT 3 - END ############################

        ############################ PT 4 - START ############################
        # # Temporal Bottleneck
        # x = self.bottleneck(x)

        # #TENSOR LIST
        # list_length = len(x)
        # print("Number of elements in the list:", list_length)
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        # # Shape of tensor 0: torch.Size([256])
        # # Shape of tensor 1: torch.Size([256])
        # # Shape of tensor 2: torch.Size([256])
        # # Shape of tensor 3: torch.Size([256])
        ############################ PT 4 - END ############################

        ############################ PT 5 - START ############################
        # RNN for Temporal Information

        # # Stack tensors along a new dimension to create a sequence
        # stacked_input = torch.stack(x, dim=0)  # This creates a tensor of shape [4, 512, 16, 16]

        # If batch_first=True, reshape to [batch_size, sequence_length, channels, height, width]
        # Here, batch_size=1 since you have one sequence of 4 frames
        # x = x.unsqueeze(0)  # Reshape to [1, 4, 512, 16, 16]

        x = x.unsqueeze(1)  # Add temporal dimension
        x = self.rnn(x)

        #TENSOR LIST
        list_length = len(x)
        print("Number of elements in the list:", list_length)
        for i, tensor in enumerate(x):
            print(f"Shape of tensor {i}: {tensor.shape}")
        # Number of elements in the list: 1
        # Shape of tensor 0: torch.Size([64, 16, 16])
        ############################ PT 5 - END ############################

        ############################ PT 6 - START ############################
        # Decoder to Regenerate Image
        x = self.decoder(x)

        #TENSOR LIST
        list_length = len(x)
        print("Number of elements in the list:", list_length)
        for i, tensor in enumerate(x):
            print(f"Shape of tensor {i}: {tensor.shape}")
        # Number of elements in the list: 1
        # Shape of tensor 0: torch.Size([3, 256, 256])
        ############################ PT 6 - END ############################

        return x
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the full pipeline with VGG16
    vgg_encoder = models.vgg16(pretrained=True)
    encoder = VGGEncoder(device, vgg_encoder.features)
    bottleneck = Bottleneck(device, 512, 256)  # Adjust the bottleneck size as needed
    rnn = VideoRNN(device, height=7, width=7, channels=512, hidden_dim=[32,64], kernel_size=(3,3), num_layers=2)  # Adjust based on your requirements
    decoder = Decoder(device, 512, 3)  # 128 is the size of the bottleneck, 3 is the number of channels in the output image
    video_pipeline = VideoAnalyticsPipeline(device, encoder, bottleneck, rnn, decoder)
    # Use GPU if available
    video_pipeline = video_pipeline.cuda()
    summary(video_pipeline, input_size =(3,256,256))

# Example usage:
# Assuming you have a video sequence tensor 'video_sequence' (batch_size, sequence_length, channels, height, width)
# output = video_pipeline(video_sequence)
