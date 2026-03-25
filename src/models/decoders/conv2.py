import torch
import torch.nn as nn

class Conv2Decoder(nn.Module):
    def __init__(self, output_channels: int, image_size: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.final_spatial_size = image_size // 4
        self.flattened_dim = (hidden_channels * 2) * (self.final_spatial_size ** 2)
        
        # Step 1: Project latent point back to the flattened convolutional size
        self.fc = nn.Linear(latent_dim, self.flattened_dim)
        
        # Step 2: Transposed Convolutions (Upsampling)
        # Doubles dimensions: 7 -> 14 or 8 -> 16
        self.conv2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        
        # Doubles dimensions again: 14 -> 28 or 16 -> 32
        self.conv1 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1)
        
        self.activation = nn.ReLU()

    def forward(self, z):
        # 1. Linear expansion
        x = self.fc(z)
        
        # 2. Reshape into a 4D tensor for convolutions: [Batch, Channels, H, W]
        x = x.view(x.size(0), self.hidden_channels * 2, self.final_spatial_size, self.final_spatial_size)
        
        # 3. Upsample layers
        x = self.activation(self.conv2(x))
        
        # 4. Final layer uses Sigmoid to squash pixel values between 0 and 1
        x = torch.sigmoid(self.conv1(x))
        return x