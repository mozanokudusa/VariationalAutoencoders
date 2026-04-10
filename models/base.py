"""
Base class for all VAE model variants.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BaseVAE(ABC, nn.Module):
    """
    Abstract base for every VAE variant in this project.

    Attributes
    ----------
    model_name : str
        Short identifier used in filenames and logs (e.g., "vae", "vq_vae").
    extra_metrics : List[str]
        Names of model-specific metrics returned by forward().
        The training loop will automatically track and log these.
        Standard metrics (loss components, PSNR, SSIM) are always tracked.
    """

    model_name: str = "base"
    extra_metrics: List[str] = []

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def encode(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Encode input into latent parameters.

        Returns a dict whose keys depend on the variant:
          - Standard VAE : {"mu": ..., "logvar": ...}
          - VQ-VAE       : {"z_e": ..., "z_q": ..., "encoding_indices": ...}
        """
        ...

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector(s) back to input space."""
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Full forward pass.

        Must return a dict containing AT LEAST:
          "recon"  : reconstructed input  (B, C, H, W)

        Plus any keys listed in `extra_metrics`, and whatever the
        loss_function needs (e.g., "mu", "logvar", "vq_loss").
        """
        ...

    @abstractmethod
    def loss_function(
        self, model_output: Dict[str, Tensor], target: Tensor, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Compute all loss components.

        Parameters
        ----------
        model_output : dict returned by forward()
        target       : original input tensor  (B, C, H, W)

        Returns
        -------
        dict with AT LEAST:
          "total_loss" : scalar tensor used for backprop
        Plus any named components (e.g., "recon_loss", "kl_loss").
        All values should be scalar tensors.
        """
        ...

    @abstractmethod
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """
        Generate new samples from the prior.

        Returns tensor of shape (num_samples, C, H, W).
        """
        ...

    def reconstruct(self, x: Tensor) -> Tensor:
        """Return only the reconstruction (no grad by default)."""
        with torch.no_grad():
            return self.forward(x)["recon"]

    def get_latent_dim(self) -> int:
        """Return dimensionality of the latent space (override if needed)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented get_latent_dim()"
        )


MODEL_REGISTRY: Dict[str, type] = {}


def register_model(key: str, cls: type) -> None:
    """Register a model class under the given key."""
    if not issubclass(cls, BaseVAE):
        raise TypeError(f"{cls} does not inherit from BaseVAE")
    MODEL_REGISTRY[key] = cls


def get_model(key: str, **kwargs) -> BaseVAE:
    """Instantiate a registered model by its key."""
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys()) or "(none)"
        raise KeyError(
            f"Unknown model '{key}'. Available: {available}"
        )
    return MODEL_REGISTRY[key](**kwargs)
