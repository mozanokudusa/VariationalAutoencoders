import torch
import torch.nn as nn

class Conv3Encoder_Batch(nn.Module):
    def __init__(self, input_channels, image_size, hidden_channels, latent_dim):
        super().__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # Layer 2
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        
        # Layer 3
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels * 4)
        
        self.final_spatial = 4
        self.flattened_dim = (hidden_channels * 4) * (self.final_spatial ** 2)
        
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        
        # LeakyReLU with a small negative slope
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)