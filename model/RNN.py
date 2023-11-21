import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Fully connected layer to map RNN output to desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = torch.stack(x , dim=2)

        # Forward pass through the RNN layer
        out, _ = self.rnn(x)

        # Only take the output from the final time step
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