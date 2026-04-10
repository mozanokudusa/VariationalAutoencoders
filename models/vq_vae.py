"""
Vector-Quantised Variational Autoencoder (VQ-VAE).
=====================================================================
Architecture (mirrors conv_vae.py structure):
  Encoder : (B,C,H,W) -> [Conv-BN-ReLU stride-2] x N -> Conv1x1 -> (B, embed_dim, H', W')
  Quantizer: snap each spatial vector to nearest codebook entry
  Decoder : (B, embed_dim, H', W') -> [ConvTranspose-BN-ReLU stride-2] x N -> (B,C,H,W)
=====================================================================
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseVAE, register_model


# Vector Quantizer (EMA + standard gradient modes)
class VectorQuantizer(nn.Module):
    """
    Quantises each spatial position of the input feature map to the
    nearest codebook vector.

    Parameters
    ----------
    num_embeddings  : int   – codebook size K
    embedding_dim   : int   – dimension of each codebook vector
    commitment_cost : float – beta weight on ||z_e - sg[z_q]||^2
    use_ema         : bool  – update codebook with EMA (recommended)
    ema_decay       : float – decay for EMA updates
    ema_epsilon     : float – Laplace smoothing for EMA cluster sizes
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema

        # Codebook: K vectors of dimension D
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        if use_ema:
            self.embedding.weight.requires_grad = False
            self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer(
                "_ema_embed_sum",
                self.embedding.weight.clone(),
            )
            self._ema_decay = ema_decay
            self._ema_epsilon = ema_epsilon

    def forward(self, z_e: Tensor) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        z_e : (B, D, H', W') – continuous encoder output

        Returns
        -------
        dict with:
          z_q              : (B, D, H', W') – quantised, straight-through gradient
          vq_loss          : scalar – vector quantisation loss
          commitment_loss  : scalar – commitment loss (weighted)
          encoding_indices : (B, H', W') – codebook index per spatial position
          codebook_usage   : scalar – fraction of codebook entries used in this batch
          perplexity       : scalar – exp(entropy) of codebook usage distribution
        """

        with torch.amp.autocast('cuda', enabled=False):
            z_e = z_e.float()

            B, D, H, W = z_e.shape

            z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D) 

            # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
            codebook = self.embedding.weight  # (K, D)
            distances = (
                z_flat.pow(2).sum(dim=1, keepdim=True)        
                + codebook.pow(2).sum(dim=1, keepdim=False)   
                - 2.0 * z_flat @ codebook.t()                 
            )

            # Nearest codebook entry for each spatial position
            encoding_indices = distances.argmin(dim=1)  

            # Quantised vectors
            z_q_flat = self.embedding(encoding_indices)  
            z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

            # Losses
            if self.use_ema and self.training:
                encodings_onehot = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).float()

                self._ema_cluster_size.mul_(self._ema_decay).add_(
                    encodings_onehot.sum(dim=0), alpha=1.0 - self._ema_decay
                )
                embed_sum = encodings_onehot.t() @ z_flat.detach()
                self._ema_embed_sum.mul_(self._ema_decay).add_(
                    embed_sum, alpha=1.0 - self._ema_decay
                )

                # Laplace smoothing
                n = self._ema_cluster_size.sum()
                cluster_size = (
                    (self._ema_cluster_size + self._ema_epsilon)
                    / (n + self.num_embeddings * self._ema_epsilon)
                    * n
                )
                self.embedding.weight.data.copy_(
                    self._ema_embed_sum / cluster_size.unsqueeze(1)
                )

                vq_loss = torch.tensor(0.0, device=z_e.device)
            else:
                # Standard VQ loss: ||sg[z_e] - e||^2
                vq_loss = F.mse_loss(z_q, z_e.detach())

            # Commitment loss: ||z_e - sg[z_q]||^2
            commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())

            # Straight-through estimator
            z_q_st = z_e + (z_q - z_e).detach()

            # Codebook usage metrics
            unique_codes = encoding_indices.unique().numel()
            codebook_usage = unique_codes / self.num_embeddings

            # Perplexity: exp(H) where H = entropy of usage distribution
            encodings_onehot = F.one_hot(
                encoding_indices, self.num_embeddings
            ).float()
            avg_probs = encodings_onehot.mean(dim=0)  # (K,)
            perplexity = torch.exp(
                -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            )

            return {
                "z_q": z_q_st,
                "vq_loss": vq_loss,
                "commitment_loss": commitment_loss,
                "encoding_indices": encoding_indices.view(B, H, W),
                "codebook_usage": codebook_usage,
                "perplexity": perplexity,
            }



# Encoder
class VQEncoder(nn.Module):
    """
    Strided-conv encoder that maps (B, C, H, W) → (B, embedding_dim, H', W').

    Unlike ConvEncoder in conv_vae.py, there is no flatten or FC layer.
    A final 1×1 conv projects the last hidden channel count down to
    embedding_dim to match the codebook vector dimension.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int],
        embedding_dim: int,
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

        # Project
        self.proj = nn.Conv2d(in_ch, embedding_dim, kernel_size=1)

        # Compute bottleneck spatial size
        spatial = image_size
        for _ in hidden_dims:
            spatial = (spatial + 1) // 2
        self.spatial = spatial

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_backbone(x)
        return self.proj(h)


# Decoder
class VQDecoder(nn.Module):
    """
    Transposed-conv decoder that maps (B, embedding_dim, H', W') → (B, C, H, W).

    No FC layer — works entirely in spatial feature-map space.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int],
        output_channels: int,
    ) -> None:
        super().__init__()

        # Project
        first_ch = hidden_dims[0] if hidden_dims else embedding_dim
        self.input_proj = nn.Conv2d(embedding_dim, first_ch, kernel_size=1)

        # Transposed conv base
        layers: List[nn.Module] = []
        in_ch = first_ch
        for out_ch in hidden_dims[1:]:
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

        # Final projection to image channels
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, output_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z_q: Tensor) -> Tensor:
        h = self.input_proj(z_q)
        h = self.deconv_backbone(h)
        return self.final_conv(h)


# Full VQ-VAE
class VQVAE(BaseVAE):
    """
    VQ-VAE combining VQEncoder + VectorQuantizer + VQDecoder.

    Parameters
    ----------
    input_channels  : int        – image channels (1 or 3)
    image_size      : int        – spatial resolution (square, default 32)
    hidden_dims     : List[int]  – encoder conv channel counts
    latent_dim      : int        – embedding dimension of each codebook vector
    num_embeddings  : int        – codebook size K (default 512)
    commitment_cost : float      – beta for commitment loss (default 0.25)
    use_ema         : bool       – EMA codebook updates (default True)
    ema_decay       : float      – EMA decay rate (default 0.99)
    recon_loss_type : str        – "bce" or "mse"
    kl_weight       : float      – ignored (accepted for get_model() compat)
    """

    model_name: str = "vq_vae"
    extra_metrics: List[str] = ["codebook_usage", "perplexity"]

    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 32,
        hidden_dims: List[int] | None = None,
        latent_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        recon_loss_type: str = "bce",
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.input_channels = input_channels
        self.image_size = image_size
        self.embedding_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.recon_loss_type = recon_loss_type
        self.hidden_dims = list(hidden_dims)

        # Encoder
        self.encoder = VQEncoder(
            input_channels, hidden_dims, latent_dim, image_size
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )

        # Decoder
        # The decoder needs len(hidden_dims) upsample steps to match the encoder's downsample steps.
        decoder_hidden = list(reversed(hidden_dims))

        self.decoder = VQDecoder(
            embedding_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_channels=input_channels,
        )

    def encode(self, x: Tensor) -> Dict[str, Tensor]:
        z_e = self.encoder(x)  # (B, D, H', W')
        vq_out = self.quantizer(z_e)
        return {
            "z_e": z_e,
            "z_q": vq_out["z_q"],
            "encoding_indices": vq_out["encoding_indices"],
            "vq_loss": vq_out["vq_loss"],
            "commitment_loss": vq_out["commitment_loss"],
            "codebook_usage": vq_out["codebook_usage"],
            "perplexity": vq_out["perplexity"],
        }

    def decode(self, z: Tensor) -> Tensor:
        raw = self.decoder(z)
        return raw[:, :, :self.image_size, :self.image_size]

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        enc_out = self.encode(x)
        recon = self.decode(enc_out["z_q"])
        return {
            "recon": recon,
            "z_e": enc_out["z_e"],
            "z_q": enc_out["z_q"],
            "vq_loss": enc_out["vq_loss"],
            "commitment_loss": enc_out["commitment_loss"],
            "encoding_indices": enc_out["encoding_indices"],
            "codebook_usage": enc_out["codebook_usage"],
            "perplexity": enc_out["perplexity"],
        }

    def loss_function(
        self, model_output: Dict[str, Tensor], target: Tensor, **kwargs
    ) -> Dict[str, Tensor]:
        recon = model_output["recon"]
        vq_loss = model_output["vq_loss"]
        commitment_loss = model_output["commitment_loss"]

        # Reconstruction loss
        if self.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy(
                recon, target, reduction="sum"
            ) / target.size(0)
        else:
            recon_loss = F.mse_loss(
                recon, target, reduction="sum"
            ) / target.size(0)

        total_loss = recon_loss + vq_loss + commitment_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
        }

    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """
        Draw random codebook entries — produces incoherent images.
        For proper generation, train a prior (e.g., PixelCNN) over
        the discrete code grid and sample from that instead.
        """
        H = W = self.encoder.spatial
        random_indices = torch.randint(
            0, self.num_embeddings, (num_samples, H, W), device=device
        )
        z_q = self.quantizer.embedding(random_indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)

    def get_latent_dim(self) -> int:
        return self.embedding_dim


register_model("vq_vae", VQVAE)
