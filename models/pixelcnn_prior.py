"""
PixelCNN Prior for VQ-VAE.
=====================================================================
Learns the distribution over the discrete code grid produced by a
trained VQ-VAE encoder.  Used to generate coherent new images by
sampling codebook indices autoregressively, then decoding with the
VQ-VAE decoder.

Architecture:
  Input  : (B, 1, H', W') grid of codebook indices → embedding →
           (B, D, H', W')
  Body   : MaskedConv2d (type A) → [MaskedConv2d (type B) + ReLU] x N
  Output : (B, K, H', W') logits over K codebook entries per position

Masked convolutions enforce the autoregressive ordering: each
position can only see positions above it and to the left (raster
scan order).
=====================================================================
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Masked Convolution
class MaskedConv2d(nn.Conv2d):
    """
    Conv2d with a binary mask enforcing autoregressive ordering.

    Parameters
    ----------
    mask_type : str
        'A' – the center pixel is also masked (first layer only).
        'B' – the center pixel is visible (all subsequent layers).
    """

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        assert mask_type in ("A", "B")
        # Force odd kernel, same-padding
        padding = kernel_size // 2
        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, **kwargs,
        )

        # Build mask: 1 = visible, 0 = masked
        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, kH, kW = self.weight.shape
        center_h, center_w = kH // 2, kW // 2

        # Zero out everything below the center row
        self.mask[:, :, center_h + 1:, :] = 0
        # Zero out everything to the right of center on the center row
        self.mask[:, :, center_h, center_w + 1:] = 0

        if mask_type == "A":
            # Also mask the center pixel itself
            self.mask[:, :, center_h, center_w] = 0

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)



# Masked Conv Residual Block

class MaskedResidualBlock(nn.Module):
    """Residual block using type-B masked convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", channels, channels, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d("B", channels, channels, kernel_size=kernel_size),
            nn.ReLU(),
            MaskedConv2d("B", channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


# PixelCNN Prior
class PixelCNNPrior(nn.Module):
    """
    Autoregressive prior over a grid of discrete codebook indices.

    Parameters
    ----------
    num_embeddings : int  – codebook size K (vocabulary)
    grid_size      : int  – spatial size of the code grid (H' = W')
    hidden_dim     : int  – channel width of the internal conv layers
    num_layers     : int  – number of masked residual blocks
    embedding_dim  : int  – dimension of the input embedding per code
    kernel_size    : int  – kernel size for masked convolutions
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        grid_size: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 8,
        embedding_dim: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.grid_size = grid_size

        # Embed discrete codes into continuous vectors
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # First layer
        self.input_conv = MaskedConv2d(
            "A", embedding_dim, hidden_dim, kernel_size=kernel_size
        )

        # Residual stack
        self.res_blocks = nn.Sequential(
            *[MaskedResidualBlock(hidden_dim, kernel_size) for _ in range(num_layers)]
        )

        # Output projection:
        self.output_net = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1),
        )

    def forward(self, codes: Tensor) -> Tensor:
        """
        Parameters
        ----------
        codes : (B, H', W') LongTensor of codebook indices

        Returns
        -------
        logits : (B, K, H', W') – unnormalized log-probabilities
        """
        x = self.embedding(codes).permute(0, 3, 1, 2).contiguous()

        x = self.input_conv(x)
        x = self.res_blocks(x)
        logits = self.output_net(x)

        return logits

    def loss(self, codes: Tensor) -> Tensor:
        """
        Cross-entropy loss for training.

        Parameters
        ----------
        codes : (B, H', W') LongTensor of codebook indices

        Returns
        -------
        scalar loss (mean over batch and spatial positions)
        """
        logits = self.forward(codes)
        return F.cross_entropy(logits, codes)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressively sample code grids.

        Parameters
        ----------
        num_samples : int
        device      : torch.device
        temperature : float – controls randomness (lower = more greedy)
        top_k       : int   – if set, restrict sampling to top-k logits

        Returns
        -------
        codes : (num_samples, H', W') LongTensor of sampled indices
        """
        H = W = self.grid_size
        codes = torch.zeros(num_samples, H, W, dtype=torch.long, device=device)

        for i in range(H):
            for j in range(W):
                logits = self.forward(codes) 
                logits_ij = logits[:, :, i, j]  

                # Temperature scaling
                logits_ij = logits_ij / max(temperature, 1e-8)

                # Top-k filtering
                if top_k is not None and top_k < self.num_embeddings:
                    topk_vals, _ = logits_ij.topk(top_k, dim=-1)
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    logits_ij[logits_ij < threshold] = float("-inf")

                probs = F.softmax(logits_ij, dim=-1)
                codes[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)

        return codes

    @torch.no_grad()
    def decode_samples(
        self,
        vqvae_model,
        num_samples: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Sample code grids and decode them through the VQ-VAE decoder.

        Parameters
        ----------
        vqvae_model : trained VQVAE instance
        num_samples : int
        device      : torch.device
        temperature : float
        top_k       : int

        Returns
        -------
        images : (num_samples, C, H, W) tensor in [0, 1]
        """
        codes = self.sample(num_samples, device, temperature, top_k)

       
        z_q = vqvae_model.quantizer.embedding(codes)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Decode through VQ-VAE decoder
        images = vqvae_model.decode(z_q)
        return images.clamp(0, 1)
