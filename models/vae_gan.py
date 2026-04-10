"""
VAE/GAN — Variational Autoencoder with Adversarial Training.
=====================================================================
Combines a VAE (encoder + decoder) with a GAN discriminator.
The discriminator replaces pixel-wise reconstruction loss with a
learned perceptual similarity via feature matching.

Architecture:
  Encoder       : (B,C,H,W) -> [Conv-BN-ReLU stride-2] x N -> flatten -> (mu, logvar)
  Decoder/Gen   : z -> FC -> unflatten -> [ConvTranspose-BN-ReLU stride-2] x N -> (B,C,H,W)
  Discriminator : (B,C,H,W) -> [Conv-LeakyReLU stride-2] x N -> flatten -> real/fake
                  Also exposes intermediate feature maps for feature-matching loss.

Three losses, three optimizer steps per batch:
  1. L_disc    = BCE(D(real), 1) + BCE(D(recon.detach()), 0)
  2. L_enc     = kl_weight * KL + feat_weight * feature_match(real, recon)
  3. L_dec     = feat_weight * feature_match(real, recon) + adv_weight * BCE(D(recon), 1)

This model inherits from BaseVAE for compatibility with the plotting
and metric infrastructure, but it requires a CUSTOM TRAINING LOOP
(see train/train_vaegan.ipynb) because the standard train_one_epoch
cannot handle three optimizers.

Reference:
  Larsen et al., "Autoencoding beyond pixels using a learned
  similarity metric" (ICML 2016)  https://arxiv.org/abs/1512.09300
=====================================================================
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseVAE, register_model


# ------------------------------------------------------------------
# Encoder (same as ConvVAE)
# ------------------------------------------------------------------
class VAEGANEncoder(nn.Module):
    """Strided-conv encoder: (B,C,H,W) → (mu, logvar)."""

    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int],
        latent_dim: int,
        image_size: int = 32,
    ) -> None:
        super().__init__()

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

        spatial = image_size
        for _ in hidden_dims:
            spatial = (spatial + 1) // 2
        self.flat_size = hidden_dims[-1] * spatial * spatial
        self.spatial = spatial
        self.last_channels = hidden_dims[-1]

        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.conv_backbone(x)
        h_flat = h.view(h.size(0), -1)
        return {"mu": self.fc_mu(h_flat), "logvar": self.fc_logvar(h_flat)}


# ------------------------------------------------------------------
# Decoder / Generator
# ------------------------------------------------------------------
class VAEGANDecoder(nn.Module):
    """Transposed-conv decoder: z → (B,C,H,W)."""

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

        self.fc = nn.Linear(latent_dim, last_enc_ch * spatial * spatial)

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


# ------------------------------------------------------------------
# Discriminator (with feature extraction)
# ------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    Convolutional discriminator that classifies real vs fake AND
    exposes intermediate feature maps for feature-matching loss.

    Uses spectral normalization for training stability.
    LeakyReLU instead of ReLU (standard for discriminators).
    No BatchNorm (interferes with spectral norm).
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int],
        image_size: int = 32,
    ) -> None:
        super().__init__()

        # Build conv blocks as separate modules so we can extract features
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        for out_ch in hidden_dims:
            block = nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.blocks.append(block)
            in_ch = out_ch

        # Compute spatial size after all stride-2 convs
        spatial = image_size
        for _ in hidden_dims:
            spatial = (spatial + 1) // 2
        flat_size = hidden_dims[-1] * spatial * spatial

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(flat_size, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Returns
        -------
        dict with:
          "logit"    : (B, 1) — raw discriminator output (pre-sigmoid)
          "features" : list of (B, C, H, W) intermediate feature maps
        """
        features = []
        h = x
        for block in self.blocks:
            h = block(h)
            features.append(h)

        logit = self.classifier(h)
        return {"logit": logit, "features": features}


# ------------------------------------------------------------------
# Full VAE/GAN
# ------------------------------------------------------------------
class VAEGAN(BaseVAE):
    """
    VAE/GAN: VAE with adversarial discriminator and feature-matching
    reconstruction loss.

    Parameters
    ----------
    input_channels  : int        – image channels (1 or 3)
    image_size      : int        – spatial resolution (square, default 32)
    hidden_dims     : List[int]  – encoder/decoder conv channel counts
    latent_dim      : int        – latent space dimensionality
    kl_weight       : float      – weight on KL divergence
    feat_weight     : float      – weight on feature-matching loss
    adv_weight      : float      – weight on adversarial loss for decoder
    disc_dims       : List[int]  – discriminator channel counts (None = same as hidden_dims)
    recon_loss_type : str        – ignored (kept for get_model() compatibility)
    """

    model_name: str = "vae_gan"
    extra_metrics: List[str] = ["disc_loss", "gen_adv_loss", "feat_match_loss"]

    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 32,
        hidden_dims: List[int] | None = None,
        latent_dim: int = 64,
        kl_weight: float = 1.0,
        feat_weight: float = 1.0,
        adv_weight: float = 1.0,
        disc_dims: List[int] | None = None,
        recon_loss_type: str = "mse",
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        if disc_dims is None:
            disc_dims = list(hidden_dims)

        self.input_channels = input_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.feat_weight = feat_weight
        self.adv_weight = adv_weight
        self.hidden_dims = list(hidden_dims)

        # Encoder
        self.encoder = VAEGANEncoder(
            input_channels, hidden_dims, latent_dim, image_size
        )

        # Decoder / Generator
        decoder_hidden = list(reversed(hidden_dims[:-1]))
        self.decoder = VAEGANDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_channels=input_channels,
            spatial=self.encoder.spatial,
            last_enc_ch=self.encoder.last_channels,
        )

        # Discriminator
        self.discriminator = Discriminator(
            input_channels, disc_dims, image_size
        )

    # ----- core interface -----

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

    def discriminate(self, x: Tensor) -> Dict[str, Tensor]:
        """Run discriminator on an image batch."""
        return self.discriminator(x)

    # ----- loss computation -----

    @staticmethod
    def feature_matching_loss(feat_real: List[Tensor], feat_fake: List[Tensor]) -> Tensor:
        """L2 distance between discriminator features of real and fake images."""
        loss = torch.tensor(0.0, device=feat_real[0].device)
        for fr, ff in zip(feat_real, feat_fake):
            loss = loss + F.mse_loss(ff, fr.detach())
        return loss / len(feat_real)

    def compute_disc_loss(
        self, x_real: Tensor, x_recon: Tensor
    ) -> Dict[str, Tensor]:
        """
        Discriminator loss: classify real as 1, reconstructed as 0.
        """
        disc_real = self.discriminate(x_real)
        disc_fake = self.discriminate(x_recon.detach())  # detach decoder

        real_labels = torch.ones_like(disc_real["logit"])
        fake_labels = torch.zeros_like(disc_fake["logit"])

        loss_real = F.binary_cross_entropy_with_logits(disc_real["logit"], real_labels)
        loss_fake = F.binary_cross_entropy_with_logits(disc_fake["logit"], fake_labels)
        disc_loss = 0.5 * (loss_real + loss_fake)

        return {
            "disc_loss": disc_loss,
            "disc_real_acc": (disc_real["logit"] > 0).float().mean(),
            "disc_fake_acc": (disc_fake["logit"] < 0).float().mean(),
        }

    def compute_enc_loss(
        self, mu: Tensor, logvar: Tensor,
        feat_real: List[Tensor], feat_recon: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Encoder loss: KL divergence + feature-matching reconstruction.
        """
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / mu.size(0)

        feat_loss = self.feature_matching_loss(feat_real, feat_recon)

        enc_loss = self.kl_weight * kl_loss + self.feat_weight * feat_loss

        return {
            "enc_loss": enc_loss,
            "kl_loss": kl_loss,
            "feat_match_loss": feat_loss,
        }

    def compute_dec_loss(
        self, x_recon: Tensor,
        feat_real: List[Tensor], feat_recon: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Decoder/Generator loss: feature matching + adversarial (fool disc).
        """
        feat_loss = self.feature_matching_loss(feat_real, feat_recon)

        disc_fake = self.discriminate(x_recon)
        real_labels = torch.ones_like(disc_fake["logit"])
        adv_loss = F.binary_cross_entropy_with_logits(disc_fake["logit"], real_labels)

        dec_loss = self.feat_weight * feat_loss + self.adv_weight * adv_loss

        return {
            "dec_loss": dec_loss,
            "gen_adv_loss": adv_loss,
        }

    def loss_function(
        self, model_output: Dict[str, Tensor], target: Tensor, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Combined loss for logging compatibility.
        NOTE: This is called AFTER the three-step update in the training loop.
        It recomputes losses for metric tracking only (no backprop through this).
        """
        recon = model_output["recon"]
        mu = model_output["mu"]
        logvar = model_output["logvar"]

        # Pixel-wise recon loss (for PSNR/SSIM tracking — not used in training)
        recon_loss = F.mse_loss(recon, target, reduction="sum") / target.size(0)

        # KL
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


# ------------------------------------------------------------------
# Register
# ------------------------------------------------------------------
register_model("vae_gan", VAEGAN)
