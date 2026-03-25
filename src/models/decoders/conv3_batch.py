import torch
import torch.nn as nn

class Conv3Decoder(nn.Module):
    def __init__(self, output_channels, image_size, hidden_channels, latent_dim):
        super().__init__()
        self.final_spatial = 4
        self.hidden_channels = hidden_channels
        self.flattened_dim = (hidden_channels * 4) * (self.final_spatial ** 2)
        
        self.fc = nn.Linear(latent_dim, self.flattened_dim)
        
        # Layers
        self.conv3 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels * 2)
        
        self.conv2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv1 = nn.ConvTranspose2d(hidden_channels, output_channels, 4, stride=2, padding=1)
        
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_channels * 4, self.final_spatial, self.final_spatial)
        
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        # No BatchNorm on the final layer to preserve color range
        return torch.sigmoid(self.conv1(x))