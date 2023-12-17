import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batchnorm=False):
        super(SimpleRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out, _ = self.rnn(x)

        if self.batchnorm:
            out = self.batchnorm_layer(out)

        out = self.fc(out[:, -1, :])

        return out
    
class MultiLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MultiLayerRNN, self).__init__()

        # Define the multi-layer RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer to map the final RNN output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = torch.stack(x , dim=2)

        # Forward pass through the multi-layer RNN
        out, _ = self.rnn(x)

        # Only take the output from the final time step
        out = self.fc(out[:, -1, :])

        return out
    
class ExtendedSimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ExtendedSimpleRNN, self).__init__()

        # Additional MLP for hidden-to-hidden connections
        self.mlp_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # You can customize the size and number of hidden layers
            nn.ReLU(),
        )

        # Original RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Fully connected layer to map RNN output to desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = torch.stack(x , dim=2)
        # Forward pass through the additional MLP for hidden-to-hidden connections
        x = self.mlp_hidden(x)

        # Forward pass through the RNN layer
        out, _ = self.rnn(x)

        # Only take the output from the final time step
        out = self.fc(out[:, -1, :])

        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batchnorm=False):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out, _ = self.lstm(x)

        if self.batchnorm:
            out = self.batchnorm_layer(out)

        out = self.fc(out[:, -1, :])

        return out

class CNN_RNNModel(nn.Module):
    def __init__(self, cnn_input_channels, cnn_output_channels, rnn_input_size, rnn_hidden_size, rnn_output_size, num_rnn_layers, batchnorm=False):
        super(CNN_RNNModel, self).__init__()

        # Convolutional Neural Network (CNN) module
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_channels, cnn_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Simple Recurrent Neural Network (RNN) module
        self.rnn = nn.RNN(rnn_input_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, rnn_output_size)

        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(rnn_hidden_size)

    def forward(self, images, sequence):
        # CNN forward pass
        cnn_output = self.cnn(images)

        # Reshape CNN output for compatibility with RNN
        cnn_output = cnn_output.view(cnn_output.size(0), -1)

        # Concatenate CNN output with RNN input (sequence)
        combined_input = torch.cat((cnn_output, sequence), dim=1)

        # RNN forward pass
        rnn_output, _ = self.rnn(combined_input.unsqueeze(1))  # Add a time dimension

        if self.batchnorm:
            rnn_output = self.batchnorm_layer(rnn_output.squeeze(1))

        # Fully connected layer
        output = self.fc(rnn_output[:, -1, :])

        return output
