import torch
import torch.nn as nn

class Conv3Encoder(nn.Module):
    def __init__(self, input_channels: int, image_size: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Layer 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 4, stride=2, padding=1)
        # Layer 2: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1)
        # Layer 3: 8x8 -> 4x4
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1)
        
        # 32 / 2 / 2 / 2 = 4
        self.final_spatial_size = 4 
        self.flattened_dim = (hidden_channels * 4) * (self.final_spatial_size ** 2)
        
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)