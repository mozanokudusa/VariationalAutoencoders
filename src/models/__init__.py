import torch
import torch.nn as nn

# Import Standard VAE Components
from .encoders.conv2 import Conv2Encoder
from .encoders.conv3 import Conv3Encoder
from .decoders.conv2 import Conv2Decoder
from .decoders.conv3 import Conv3Decoder

# Import VQ-VAE Assembly
from .vq_vae import VQVAE

# Registries for standard VAE modularity
ENCODER_REGISTRY = {
    "conv2": Conv2Encoder,
    "conv3": Conv3Encoder
}

DECODER_REGISTRY = {
    "conv2": Conv2Decoder,
    "conv3": Conv3Decoder
}

class StandardVAE(nn.Module):
    """Wrapper to assemble standard encoders and decoders."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def get_model(config):
    """Factory function to build models based on the YAML config."""
    model_config = config['model']
    model_name = model_config.get('name', 'variational_autoencoder')

    if model_name == 'variational_autoencoder':
        # 1. Build Standard VAE
        enc_type = model_config.get('encoder_type', 'conv2')
        dec_type = model_config.get('decoder_type', 'conv2')
        
        encoder_class = ENCODER_REGISTRY[enc_type]
        decoder_class = DECODER_REGISTRY[dec_type]
        
        encoder = encoder_class(
            input_channels=model_config['input_channels'],
            image_size=model_config['image_size'],
            hidden_channels=model_config['capacity'],
            latent_dim=model_config['latent_dims']
        )
        
        decoder = decoder_class(
            output_channels=model_config['input_channels'],
            image_size=model_config['image_size'],
            hidden_channels=model_config['capacity'],
            latent_dim=model_config['latent_dims']
        )
        
        return StandardVAE(encoder, decoder)

    elif model_name == 'vq_vae':
        # 2. Build VQ-VAE
        return VQVAE(
            input_channels=model_config['input_channels'],
            #image_size=model_config['image_size'],
            hidden_channels=model_config['capacity'],
            num_embeddings=model_config['num_embeddings'],
            embedding_dim=model_config['embedding_dim'],
            commitment_cost=model_config.get('commitment_cost', 0.25)
        )

    else:
        raise ValueError(f"Unknown model name in config: {model_name}")