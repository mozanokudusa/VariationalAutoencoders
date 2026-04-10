"""
Convolutional Variational Autoencoder (ConvVAE).
=====================================================================
Architecture:
  Encoder : (B,C,H,W) -> [Conv-BN-ReLU stride-2] x N -> flatten -> (mu, logvar)
  Decoder : z -> FC -> unflatten -> [ConvTranspose-BN-ReLU stride-2] x N -> (B,C,H,W)

Each entry in `hidden_dims` defines one strided conv block, so the
spatial resolution halves per layer.  For 32×32 input:
  hidden_dims = [32, 64]              → bottleneck 8×8
  hidden_dims = [32, 64, 128]         → bottleneck 4×4
  hidden_dims = [32, 64, 128, 256]    → bottleneck 2×2  (default)

The parameter name `hidden_dims` is shared with the linear VAE so
that `get_model(key, hidden_dims=...)` works for both.
=====================================================================
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseVAE, register_model


# Encoder
class ConvEncoder(nn.Module):
    """
    Strided-conv encoder that maps (B, C, H, W) → (mu, logvar).

    Parameters
    ----------
    input_channels : int        – image channels (1 or 3)
    hidden_dims    : List[int]  – output channels per conv block
    latent_dim     : int        – latent vector length
    image_size     : int        – spatial resolution of the input (square)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int],
        latent_dim: int,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        # Conv Base
        layers: List[nn.Module] = []
        in_ch = input_channels
        for out_ch in hidden_dims:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*layers)

        # Each stride-2 conv halves spatial dims: H_out = ceil(H_in / 2)
        spatial = image_size
        for _ in hidden_dims:
            spatial = (spatial + 1) // 2  # equivalent to ceil(spatial / 2)
        self.flat_size = hidden_dims[-1] * spatial * spatial
        self.spatial = spatial
        self.last_channels = hidden_dims[-1]

        # Projection
        self.fc_mu     = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.conv_backbone(x)
        h_flat = h.view(h.size(0), -1)
        return {"mu": self.fc_mu(h_flat), "logvar": self.fc_logvar(h_flat)}



# Decoder
class ConvDecoder(nn.Module):
    """
    Transposed-conv decoder that maps z → (B, C, H, W).

    Parameters
    ----------
    latent_dim     : int        – latent vector length
    hidden_dims    : List[int]  – channel counts in *decoder* order
                                  (reversed relative to encoder)
    output_channels: int        – image channels to reconstruct
    spatial        : int        – spatial size at the bottleneck
    last_enc_ch    : int        – channel depth at the bottleneck
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_channels: int,
        spatial: int,
        last_enc_ch: int,
    ) -> None:
        super().__init__()

        self.spatial = spatial
        self.last_enc_ch = last_enc_ch

        # FC from latent to spatial feature map
        self.fc = nn.Linear(latent_dim, last_enc_ch * spatial * spatial)

        # Transposed Base
        layers: List[nn.Module] = []
        in_ch = last_enc_ch
        for out_ch in hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.deconv_backbone = nn.Sequential(*layers)

        # Final Projection
        self.final_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_ch, output_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z)
        h = h.view(-1, self.last_enc_ch, self.spatial, self.spatial)
        h = self.deconv_backbone(h)
        return self.final_conv(h)


# Full Convolutional VAE
class ConvVariationalAutoencoder(BaseVAE):
    """
    Convolutional VAE combining ConvEncoder + ConvDecoder with the
    reparameterization trick.

    Parameters
    ----------
    input_channels  : int        – image channels (1 or 3)
    image_size      : int        – spatial resolution (square, default 32)
    hidden_dims     : List[int]  – encoder conv channel counts
                                   (decoder mirrors in reverse)
    latent_dim      : int        – latent space dimensionality
    kl_weight       : float      – beta weight on KL (beta-VAE when > 1)
    recon_loss_type : str        – "bce" or "mse"
    """

    model_name: str = "conv_vae"
    extra_metrics: List[str] = []

    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 32,
        hidden_dims: List[int] | None = None,
        latent_dim: int = 64,
        kl_weight: float = 1.0,
        recon_loss_type: str = "bce",
        **kwargs,  # absorb model-specific params (e.g., VQ-VAE fields)
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.input_channels = input_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.recon_loss_type = recon_loss_type
        self.hidden_dims = list(hidden_dims)

        # Encoder
        self.encoder = ConvEncoder(
            input_channels, hidden_dims, latent_dim, image_size
        )

        # Decoder
        decoder_hidden = list(reversed(hidden_dims[:-1]))
        # decoder_hidden e.g. [128, 64, 32] when encoder is [32,64,128,256]

        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_channels=input_channels,
            spatial=self.encoder.spatial,
            last_enc_ch=self.encoder.last_channels,
        )

    # core
    def encode(self, x: Tensor) -> Dict[str, Tensor]:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        raw = self.decoder(z)
        return raw[:, :, :self.image_size, :self.image_size]

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        enc_out = self.encode(x)
        mu, logvar = enc_out["mu"], enc_out["logvar"]
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def loss_function(
        self, model_output: Dict[str, Tensor], target: Tensor, **kwargs
    ) -> Dict[str, Tensor]:
        recon  = model_output["recon"]
        mu     = model_output["mu"]
        logvar = model_output["logvar"]

        # Reconstruction loss
        if self.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy(
                recon, target, reduction="sum"
            ) / target.size(0)
        else:
            recon_loss = F.mse_loss(
                recon, target, reduction="sum"
            ) / target.size(0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / target.size(0)

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def get_latent_dim(self) -> int:
        return self.latent_dim


register_model("conv_vae", ConvVariationalAutoencoder)
