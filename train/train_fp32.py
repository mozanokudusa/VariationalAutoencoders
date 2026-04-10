"""
Training script – standard FP32 precision.
=====================================================================
This script is model-agnostic: it works with any class registered in
MODEL_REGISTRY that inherits from BaseVAE.

CHANGE POINTS for a new training variant (e.g., mixed precision):
  1. Copy this file to train/train_mixed.py.
  2. Wrap the forward + loss in torch.cuda.amp.autocast().
  3. Replace optimizer.step() with scaler.step(optimizer).
  4. Set run_log.precision = "mixed".
  (All other infrastructure stays identical.)

CHANGE POINTS for a new model:
  • Nothing here.  Just pass --model <key> matching the registry.

Usage examples:
  python -m train.train_fp32 --model vae --dataset mnist --epochs 50
  python -m train.train_fp32 --model vae --dataset cifar10 --latent_dim 128 --kl_weight 0.5
=====================================================================
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# --------------- project imports ---------------
# Ensure project root is on sys.path so imports work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import get_dataloaders, IMAGE_SIZE
from models.base import get_model, MODEL_REGISTRY
# Import model modules so they register themselves
import models.vae  # noqa: F401  ← CHANGE POINT: import new model files here

from train.train_utils import (
    MetricTracker,
    RunLog,
    Timer,
    compute_image_metrics,
    save_checkpoint,
)


# ==================================================================
# Argument parsing
# ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a VAE variant (FP32)")

    # Model
    p.add_argument("--model", type=str, default="vae",
                    choices=list(MODEL_REGISTRY.keys()),
                    help="Registered model key")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256],
                    help="Encoder hidden layer widths (decoder mirrors)")
    p.add_argument("--kl_weight", type=float, default=1.0,
                    help="Beta weight on KL divergence (beta-VAE)")
    p.add_argument("--recon_loss", type=str, default="bce",
                    choices=["bce", "mse"])

    # Data
    p.add_argument("--dataset", type=str, default="mnist",
                    choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_root", type=str, default="./data/raw")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler", type=str, default="none",
                    choices=["none", "cosine", "step"])
    p.add_argument("--step_size", type=int, default=20,
                    help="StepLR step size (only if --scheduler step)")
    p.add_argument("--gamma", type=float, default=0.5,
                    help="StepLR gamma (only if --scheduler step)")

    # Checkpointing / logging
    p.add_argument("--checkpoint_dir", type=str, default="./models/checkpoints")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--save_every", type=int, default=10,
                    help="Save checkpoint every N epochs (0 = only best & final)")
    p.add_argument("--run_name", type=str, default=None,
                    help="Custom run name (auto-generated if omitted)")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                    help="Force device (auto-detected if omitted)")

    return p.parse_args()


# ==================================================================
# Training loop
# ==================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tracker: MetricTracker,
) -> None:
    """Run one training epoch."""
    model.train()
    for batch_data, _ in loader:
        batch_data = batch_data.to(device)
        batch_size = batch_data.size(0)

        # Forward
        model_output = model.forward(batch_data)
        loss_dict = model.loss_function(model_output, batch_data)
        total_loss = loss_dict["total_loss"]

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            recon = model_output["recon"]
            img_metrics = compute_image_metrics(recon, batch_data)

        # Extract model-specific extras from forward output
        extra_dict = {
            k: model_output[k]
            for k in model.extra_metrics
            if k in model_output
        }

        tracker.update(loss_dict, img_metrics, extra_dict, batch_size)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    tracker: MetricTracker,
) -> None:
    """Run evaluation on test/validation set."""
    model.eval()
    for batch_data, _ in loader:
        batch_data = batch_data.to(device)
        batch_size = batch_data.size(0)

        model_output = model.forward(batch_data)
        loss_dict = model.loss_function(model_output, batch_data)

        recon = model_output["recon"]
        img_metrics = compute_image_metrics(recon, batch_data)

        extra_dict = {
            k: model_output[k]
            for k in model.extra_metrics
            if k in model_output
        }

        tracker.update(loss_dict, img_metrics, extra_dict, batch_size)


# ==================================================================
# Main
# ==================================================================

def main() -> None:
    args = parse_args()

    # ---- Seed ----
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train] Device: {device}")

    # ---- Data ----
    train_loader, test_loader, channels = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
    )
    print(f"[train] Dataset: {args.dataset}  |  channels: {channels}")

    # ---- Model ----
    model = get_model(
        args.model,
        input_channels=channels,
        image_size=IMAGE_SIZE,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        recon_loss_type=args.recon_loss,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model: {model.model_name}  |  params: {total_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Scheduler ----
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    # ---- Run log ----
    run_name = args.run_name or (
        f"{model.model_name}_{args.dataset}_z{args.latent_dim}_fp32"
    )
    run_log = RunLog(
        model_name=model.model_name,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        optimizer_name="Adam",
        scheduler_name=args.scheduler,
        precision="fp32",
        device=str(device),
        seed=args.seed,
        extra_hparams={
            "hidden_dims": args.hidden_dims,
            "recon_loss": args.recon_loss,
            "weight_decay": args.weight_decay,
        },
    )

    log_path = Path(args.log_dir) / f"{run_name}.json"
    ckpt_dir = Path(args.checkpoint_dir)

    # ---- Training ----
    timer = Timer()
    timer.start()
    best_loss = float("inf")

    print(f"[train] Starting training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        train_tracker = MetricTracker(extra_keys=model.extra_metrics)
        train_one_epoch(model, train_loader, optimizer, device, train_tracker)
        train_summary = train_tracker.summarise()

        # --- evaluate ---
        eval_tracker = MetricTracker(extra_keys=model.extra_metrics)
        evaluate(model, test_loader, device, eval_tracker)
        eval_summary = eval_tracker.summarise()

        if scheduler is not None:
            scheduler.step()

        # --- log ---
        epoch_record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_summary.items()},
            **{f"val_{k}": v for k, v in eval_summary.items()},
        }
        run_log.epoch_history.append(epoch_record)

        val_loss = eval_summary.get("total_loss", train_summary["total_loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{args.epochs}  |  "
            f"train_loss={train_summary['total_loss']:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"psnr={eval_summary.get('psnr', 0):.2f}  "
            f"ssim={eval_summary.get('ssim', 0):.4f}  "
            f"lr={lr_now:.2e}"
        )

        # --- checkpointing ---
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            run_log.best_epoch = epoch
            run_log.best_total_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, eval_summary,
                ckpt_dir / f"{run_name}_best.pt",
                scheduler=scheduler,
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, eval_summary,
                ckpt_dir / f"{run_name}_epoch{epoch}.pt",
                scheduler=scheduler,
            )

        # Periodic JSON save (crash safety)
        run_log.total_training_time_sec = timer.elapsed()
        run_log.save(log_path)

    # ---- Final ----
    run_log.total_training_time_sec = timer.elapsed()
    run_log.save(log_path)

    save_checkpoint(
        model, optimizer, args.epochs, eval_summary,
        ckpt_dir / f"{run_name}_final.pt",
        scheduler=scheduler,
    )

    print(f"\n[train] Finished in {run_log.total_training_time_sec:.1f}s")
    print(f"[train] Best val_loss={run_log.best_total_loss:.4f} at epoch {run_log.best_epoch}")
    print(f"[train] Log saved to {log_path}")
    print(f"[train] Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
