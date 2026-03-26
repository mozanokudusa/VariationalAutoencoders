import torch
import torch.nn as nn
import torch.nn.functional as F 

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices

class VQConv2Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, embedding_dim, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embedding_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        return self.activation(self.bn2(self.conv2(x)))

class VQConv2Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, output_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_channels, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, output_channels, 4, stride=2, padding=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z_q):
        x = self.activation(self.bn1(self.conv1(z_q)))
        return self.conv2(x) # Returns raw logits

class VQConv4Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim):
        super().__init__()
        
        # Layer 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # Layer 2: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        
        # Layer 3: 8x8 -> 4x4
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels * 4)
        
        # Layer 4: 4x4 -> 4x4 (No spatial reduction, just maps to embedding_dim)
        self.conv4 = nn.Conv2d(hidden_channels * 4, embedding_dim, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(embedding_dim)
        
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        return self.activation(self.bn4(self.conv4(x)))
    
class VQConv4Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, output_channels):
        super().__init__()
        
        # Layer 1: 4x4 -> 4x4
        self.conv1 = nn.Conv2d(embedding_dim, hidden_channels * 4, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels * 4)
        
        # Layer 2: 4x4 -> 8x8
        self.conv2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        
        # Layer 3: 8x8 -> 16x16
        self.conv3 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        # Layer 4: 16x16 -> 32x32 (Returns raw logits)
        self.conv4 = nn.ConvTranspose2d(hidden_channels, output_channels, 4, stride=2, padding=1)
        
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z_q):
        x = self.activation(self.bn1(self.conv1(z_q)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        return self.conv4(x)

class VQVAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.encoder = VQConv4Encoder(input_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQConv4Decoder(embedding_dim, hidden_channels, input_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        vq_loss, z_q, perplexity, _ = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity