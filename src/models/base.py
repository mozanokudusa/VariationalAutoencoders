import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    """
    The Base VAE class. 
    It orchestrates the flow: Input -> Encoder -> Latent Space -> Decoder -> Reconstruction.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The Reparameterization Trick:
        Instead of sampling z ~ N(mu, std), we sample epsilon ~ N(0, 1) 
        and calculate z = mu + epsilon * std. 
        This makes the sampling process differentiable.
        """
        if self.training:
            # Standard deviation = exp(0.5 * log_variance)
            std = torch.exp(0.5 * logvar)
            # Sample from a standard normal distribution
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            # During evaluation, we use the mean directly for a deterministic output
            return mu

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the entire VAE.
        
        Returns:
            x_recon: The reconstructed image.
            mu: The latent mean.
            logvar: The latent log-variance.
        """
        # 1. Encode the input to get distribution parameters
        mu, logvar = self.encoder(x)
        
        # 2. Sample from the distribution (latent space)
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode the latent sample back into image space
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar