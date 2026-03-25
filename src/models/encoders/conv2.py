import torch
import torch.nn as nn

class Conv2Encoder(nn.Module):
    def __init__(self, input_channels: int, image_size: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        
        # Layer 1: Halves the image dimensions (e.g., 28 -> 14 or 32 -> 16)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        
        # Layer 2: Halves dimensions again (e.g., 14 -> 7 or 16 -> 8)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1)
        
        # Dynamic Math: Calculate the flattened size for the Linear layer
        self.final_spatial_size = image_size // 4
        self.flattened_dim = (hidden_channels * 2) * (self.final_spatial_size ** 2)
        
        # Latent Space Projections
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        
        # Flatten for the linear layers: [Batch, Channels * H * W]
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar