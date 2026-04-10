"""
Training utilities – checkpointing, metrics, and JSON run logging.
=====================================================================
Handles:
  • Saving / loading model checkpoints (with optimizer & scheduler state).
  • Computing image-quality metrics (PSNR, SSIM) on [0,1]-range tensors.
  • Accumulating per-epoch metrics and flushing to a JSON log at the
    end of training.

The JSON log structure adapts automatically to whatever metrics the
model declares via its `extra_metrics` attribute – no code changes
needed when a new model variant is added.

CHANGE POINTS:
  • To add a new *standard* metric (tracked for every model), add its
    computation in `compute_image_metrics()` and initialise it in
    `MetricTracker.__init__`.
=====================================================================
"""

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ==================================================================
# Image-quality metrics
# ==================================================================

def compute_psnr(recon: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio between two batches.
    Both tensors should be in [0, max_val].
    Returns the mean PSNR across the batch.
    """
    mse = torch.mean((recon - target) ** 2, dim=[1, 2, 3])  # per-image
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(max_val**2 / mse)
    return psnr.mean().item()


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> Tensor:
    """1-D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    return g / g.sum()


def _gaussian_kernel_2d(
    size: int, sigma: float, channels: int, device: torch.device
) -> Tensor:
    """2-D Gaussian kernel for depth-wise convolution."""
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # outer product
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def compute_ssim(
    recon: Tensor,
    target: Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Structural Similarity Index between two batches.
    Returns the mean SSIM across the batch.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    channels = recon.size(1)
    device = recon.device
    pad = window_size // 2

    kernel = _gaussian_kernel_2d(window_size, 1.5, channels, device)

    mu_x = torch.nn.functional.conv2d(recon, kernel, padding=pad, groups=channels)
    mu_y = torch.nn.functional.conv2d(target, kernel, padding=pad, groups=channels)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        torch.nn.functional.conv2d(recon**2, kernel, padding=pad, groups=channels)
        - mu_x_sq
    )
    sigma_y_sq = (
        torch.nn.functional.conv2d(target**2, kernel, padding=pad, groups=channels)
        - mu_y_sq
    )
    sigma_xy = (
        torch.nn.functional.conv2d(
            recon * target, kernel, padding=pad, groups=channels
        )
        - mu_xy
    )

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def compute_image_metrics(recon: Tensor, target: Tensor) -> Dict[str, float]:
    """
    Compute all standard image-quality metrics.

    CHANGE POINT: add new standard metrics here; they will be
    automatically tracked for every model.
    """
    return {
        "psnr": compute_psnr(recon, target),
        "ssim": compute_ssim(recon, target),
    }


# ==================================================================
# Metric tracker (accumulates per-epoch values)
# ==================================================================

class MetricTracker:
    """
    Accumulates running sums for an epoch, then computes averages.

    Automatically includes whatever extra metric keys are passed in.
    Usage:
        tracker = MetricTracker(extra_keys=model.extra_metrics)
        for batch in loader:
            tracker.update(loss_dict, image_metrics, extra_dict, batch_size)
        epoch_summary = tracker.summarise()
    """

    def __init__(self, extra_keys: List[str] | None = None) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self.extra_keys = extra_keys or []

    def update(
        self,
        loss_dict: Dict[str, Any],
        image_metrics: Dict[str, float] | None = None,
        extra_dict: Dict[str, Any] | None = None,
        batch_size: int = 1,
    ) -> None:
        """Add one batch worth of metrics."""
        combined: Dict[str, float] = {}

        # Loss components (tensor -> float)
        for k, v in loss_dict.items():
            combined[k] = v.item() if isinstance(v, Tensor) else float(v)

        # Standard image metrics
        if image_metrics:
            combined.update(image_metrics)

        # Model-specific extras
        if extra_dict:
            for k in self.extra_keys:
                if k in extra_dict:
                    v = extra_dict[k]
                    combined[k] = v.item() if isinstance(v, Tensor) else float(v)

        for k, v in combined.items():
            self._sums[k] = self._sums.get(k, 0.0) + v * batch_size
            self._counts[k] = self._counts.get(k, 0) + batch_size

    def summarise(self) -> Dict[str, float]:
        """Return epoch-average for every tracked metric."""
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in sorted(self._sums)
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ==================================================================
# Checkpointing
# ==================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str | Path,
    scheduler: Optional[Any] = None,
    extra_state: Dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state:
        state.update(extra_state)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Load a checkpoint.  Returns the full state dict so the caller
    can extract `epoch`, `metrics`, or any `extra_state`.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ==================================================================
# JSON run logger
# ==================================================================

@dataclass
class RunLog:
    """
    Stores everything about a single training run.

    Serialised to JSON at the end of training (and optionally
    at each epoch for crash safety).
    """

    # --- Hyperparameters (set before training) ---
    model_name: str = ""
    dataset: str = ""
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    latent_dim: int = 0
    kl_weight: float = 1.0
    optimizer_name: str = ""
    scheduler_name: str = ""
    precision: str = "fp32"  # "fp32" | "mixed" | ...
    extra_hparams: Dict[str, Any] = field(default_factory=dict)

    # --- Populated during / after training ---
    total_training_time_sec: float = 0.0
    epoch_history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_total_loss: float = float("inf")

    # --- Device / environment ---
    device: str = ""
    seed: int = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "RunLog":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ==================================================================
# Timer helper
# ==================================================================

class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self._start: float = 0.0

    def start(self) -> None:
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start
