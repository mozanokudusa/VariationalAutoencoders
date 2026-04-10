"""
Standard Variational Autoencoder (VAE) with fully-connected layers.
=====================================================================
Architecture (linear / MLP):
  Encoder : input_dim -> h1 -> h2 -> (mu, logvar)   [latent_dim each]
  Decoder : latent_dim -> h2 -> h1 -> input_dim

CHANGE POINTS when creating a new variant:
  • Copy this file as a starting template.
  • Swap Encoder / Decoder internals (e.g., Conv layers for a ConvVAE).
  • Override loss_function() for different objectives.
  • Set `extra_metrics` for any new per-batch metrics.
  • Register the new class in models/base.py  MODEL_REGISTRY.
=====================================================================
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseVAE, register_model


# Encoder
class Encoder(nn.Module):
    """
    Maps flattened input to latent distribution parameters (mu, logvar).

    Parameters
    ----------
    input_dim   : int   – flattened input size (C * H * W)
    hidden_dims : list  – widths of hidden FC layers
    latent_dim  : int   – dimensionality of the latent space
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_features = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_features = h_dim

        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.backbone(x)
        return {"mu": self.fc_mu(h), "logvar": self.fc_logvar(h)}


# Decoder
class Decoder(nn.Module):
    """
    Maps latent vector back to flattened input space.

    Parameters
    ----------
    latent_dim  : int   – dimensionality of the latent space
    hidden_dims : list  – widths of hidden FC layers (reverse of encoder)
    output_dim  : int   – flattened output size (C * H * W)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_features = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_features = h_dim

        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# Full VAE
class VariationalAutoencoder(BaseVAE):
    """
    Standard VAE combining Encoder + Decoder with the
    reparameterization trick.

    Parameters
    ----------
    input_channels : int   – number of image channels (1 or 3)
    image_size     : int   – spatial resolution (assumes square, default 32)
    hidden_dims    : list  – encoder hidden widths; decoder mirrors them
    latent_dim     : int   – latent space dimensionality
    kl_weight      : float – beta weight on KL term (beta-VAE when > 1)
    recon_loss_type: str   – "bce" or "mse"
    """

    model_name: str = "vae"
    extra_metrics: List[str] = []  # no model-specific extras for vanilla VAE

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
            hidden_dims = [512, 256]

        self.input_channels = input_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.recon_loss_type = recon_loss_type

        input_dim = input_channels * image_size * image_size

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(
            latent_dim, list(reversed(hidden_dims)), input_dim
        )


    def encode(self, x: Tensor) -> Dict[str, Tensor]:
        flat = x.view(x.size(0), -1)
        return self.encoder(flat)

    def decode(self, z: Tensor) -> Tensor:
        flat = self.decoder(z)
        return flat.view(
            -1, self.input_channels, self.image_size, self.image_size
        )

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
        recon = model_output["recon"]
        mu = model_output["mu"]
        logvar = model_output["logvar"]

        # Reconstruction loss
        if self.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy(
                recon, target, reduction="sum"
            ) / target.size(0)
        else:
            recon_loss = F.mse_loss(recon, target, reduction="sum") / target.size(0)

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


# Register with the global model registry
register_model("vae", VariationalAutoencoder)
