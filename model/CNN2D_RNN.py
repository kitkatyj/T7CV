import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary
import numpy as np

try:
    from ConvGRU import ConvGRU
except ImportError:
    from .ConvGRU import ConvGRU

# Encoder using VGG Feature Extractor
class VGGEncoder(nn.Module):
    def __init__(self, device, original_model):
        super(VGGEncoder, self).__init__()
        # Remove the fully connected layers (classification head)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.cuda()

    def forward(self, x):
        return self.features(x)


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
    def __init__(self, device):
        super(VideoAnalyticsPipeline, self).__init__()
        vgg_encoder = models.vgg16(pretrained=True).features
        self.encoder = VGGEncoder(device, vgg_encoder)
        rnn_height, rnn_width = 16, 16  
        self.rnn = VideoRNN(device, height=rnn_height, width=rnn_width, channels=512, hidden_dim=[32,64], kernel_size=(3,3), num_layers=2)  # Adjust based on your requirements
         # Set up the full pipeline with VGG16
        self.decoder = Decoder(device, 64, 3)  # 128 is the size of the bottleneck, 3 is the number of channels in the output image
        self.cuda()

    def forward(self, x):
        # Assuming x is a list of tensors
        device = next(self.parameters()).device  # Get the device from the model's parameters
        # Move each tensor in the input list to the device
        x = [img_.cuda() for img_ in x]
        ############################ PT 1 - START ############################
        #TENSOR LIST
        # print(f"Number of elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        # # Number of elements in the list: 4
        # Shape of tensor 0: torch.Size([1, 3, 256, 256])
        # Shape of tensor 1: torch.Size([1, 3, 256, 256])
        # Shape of tensor 2: torch.Size([1, 3, 256, 256])
        # Shape of tensor 3: torch.Size([1, 3, 256, 256])
        
        ############################ PT 1 - END ############################

        # x = torch.stack(x, dim=0)  # Shape becomes [sequence_length, channels, height, width]
        # if len(x.shape) == 4:
        #     x = x.unsqueeze(0) 
        # x = x.to(device)
        # # Continue with the existing processing
        # x = self.encoder(x)
        # x = self.rnn(x)
        # x = self.decoder(x)

        ############################ PT 2 - START ############################
        #TENSOR LIST
        # Concatenate the tensors along the batch dimension
        x = torch.cat(x, dim=0)  # x is a list of tensors concatenate 
        # x = torch.stack(x, dim=1)  # x is a list of tensors concatenate 
        # print(f"after concatenate | Number of elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        # after concatenate | Number of elements in the <class 'torch.Tensor'>: 2
        # Shape of tensor 0: torch.Size([3, 256, 256])
        # Shape of tensor 1: torch.Size([3, 256, 256])
        ############################ PT 2 - END ############################

        ############################ PT 3 - START ############################
        # Spatial Encoding
        x = self.encoder(x)

        #TENSOR LIST
        # print(f"after encoder | Number of elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        # after encoder | Number of elements in the <class 'torch.Tensor'>: 2
        # Shape of tensor 0: torch.Size([512, 16, 16])
        # Shape of tensor 1: torch.Size([512, 16, 16])

        ############################ PT 3 - END ############################

        ############################ PT 5 - START ############################
        # RNN for Temporal Information

        # # Stack tensors along a new dimension to create a sequence
        # x = torch.stack(x, dim=0)  # This creates a tensor of shape [4, 512, 16, 16]
        # print(f"after stack | Number of elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")

        # If batch_first=True, reshape to [batch_size, sequence_length, channels, height, width]
        # Here, batch_size=1 since you have one sequence of 4 frames
        # x = x.unsqueeze(0)  # Reshape to [1, 4, 512, 16, 16]

        x = x.unsqueeze(0)  # Add temporal dimension
        # print(f"after unsequeeze | Number of elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        x = self.rnn(x)

        #TENSOR LIST
        # print(f"after RNN | Number of elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        # Number of elements in the list: 1
        # Shape of tensor 0: torch.Size([64, 16, 16])
        ############################ PT 5 - END ############################

        ############################ PT 6 - START ############################
        # Decoder to Regenerate Image
        x = self.decoder(x)

        #TENSOR LIST
        # print(f"Number of elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")
        # Number of elements in the list: 1
        # Shape of tensor 0: torch.Size([3, 256, 256])
        ############################ PT 6 - END ############################

        # x = x.unsqueeze(0)
        # print(f"Number of final elements in the {type(x)}:", len(x))
        # for i, tensor in enumerate(x):
        #     print(f"Shape of tensor {i}: {tensor.shape}")

        return x
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_pipeline = VideoAnalyticsPipeline(device)
    # Use GPU if available
    video_pipeline = video_pipeline.cuda()
    # summary(video_pipeline, input_size =(3,256,256))
    # summary(video_pipeline, input_size =(1,3,256,256))
   

# Example usage:
# Assuming you have a video sequence tensor 'video_sequence' (batch_size, sequence_length, channels, height, width)
# output = video_pipeline(video_sequence)
