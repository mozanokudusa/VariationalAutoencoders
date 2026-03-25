import torch
import torch.nn as nn

class Conv3Decoder(nn.Module):
    def __init__(self, output_channels: int, image_size: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.final_spatial_size = 4
        self.flattened_dim = (hidden_channels * 4) * (self.final_spatial_size ** 2)
        
        self.fc = nn.Linear(latent_dim, self.flattened_dim)
        
        # Transpose Layers: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.conv3 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(hidden_channels, output_channels, 4, stride=2, padding=1)
        
        self.activation = nn.ReLU()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_channels * 4, self.final_spatial_size, self.final_spatial_size)
        
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv2(x))
        #Sigmoid to ensure pixel values are in [0, 1]
        return torch.sigmoid(self.conv1(x))