import torch
import torch.nn as nn

class Conv2Decoder(nn.Module):
    def __init__(self, output_channels: int, image_size: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.final_spatial_size = image_size // 4
        self.flattened_dim = (hidden_channels * 2) * (self.final_spatial_size ** 2)
        
        self.fc = nn.Linear(latent_dim, self.flattened_dim)
        
        # Double dimensions: 8 -> 16
        self.conv2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        
        # Double dimensions again: 16 -> 32
        self.conv1 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1)
        
        self.activation = nn.ReLU()

    def forward(self, z):

        x = self.fc(z)
        
        #Reshape for convolutions
        x = x.view(x.size(0), self.hidden_channels * 2, self.final_spatial_size, self.final_spatial_size)
        
        x = self.activation(self.conv2(x))
        
        #Final layer used to have a Sigmoid to keep pixel values between 0 and 1 but with autocast need logits
        x = self.conv1(x)
        return x