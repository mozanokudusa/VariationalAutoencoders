import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn.functional as F

# Using the standard torch.amp for 2026
from torch.amp import autocast, GradScaler

# Custom module imports
from src.datasets import get_dataloaders
from src.models import get_model
from src.utils import set_seed, make_averager, save_checkpoint, plot_loss, refresh_bar
from src.visualizations import plot_latent_space, plot_reconstructions

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes VAE loss with BCE sum and KL Divergence.
    Returns total loss, BCE per sample, and KLD per sample.
    """
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_x.view(recon_x.size(0), -1), 
        x.view(x.size(0), -1), 
        reduction='sum'
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (recon_loss + (beta * kld)) / x.size(0)
    
    return total_loss, recon_loss / x.size(0), kld / x.size(0)

def main(config_path):
    # File Setup
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_name = os.path.basename(config_path).replace(".yaml", "")
    exp_name = f"{base_name}_mixedprecision"
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gradient Scaler for Mixed Precision
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    print(f"Starting AMP Experiment: {exp_name}")
    print(f"Device: {device} | Workers: {config['training'].get('num_workers', 0)}")

    # 2. Data and Model Initialization
    train_loader, test_loader, class_names = get_dataloaders(config)
    model = get_model(config).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 1e-5))
    )

    # 3. Robust Data Pre-fetching
    print("Pre-fetching fixed test batch...")
    try:
        test_iter = iter(test_loader)
        batch = next(test_iter)
        fixed_x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        labels = batch[1] if isinstance(batch, (list, tuple)) else torch.zeros(fixed_x.size(0))
        named_labels = [class_names[i] for i in labels]
    except Exception as e:
        print(f"Error during data pre-fetch: {e}")
        sys.exit(1)

    # Train-loop Setup
    train_loss_history = []
    num_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('kl_warmup_epochs', 0)
    final_beta = config['model'].get('variational_beta', 1.0)

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        # Calculate KL Annealing Beta
        current_beta = min(final_beta, (epoch / warmup_epochs) * final_beta) if warmup_epochs > 0 else final_beta

        model.train()
        train_averager = make_averager()
        bce_averager = make_averager()
        kld_averager = make_averager()
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            # Autocast context for FP16 operations
            with autocast('cuda', enabled=(device.type == 'cuda')):
                recon_x, mu, logvar = model(batch_x)
                loss, bce, kld = vae_loss(recon_x, batch_x, mu, logvar, beta=current_beta)
            
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            
            # Unscale and Step Optimizer
            scaler.step(optimizer)
            
            # Update scaler factor
            scaler.update()
            
            train_averager(loss.item())
            bce_averager(bce.item())
            kld_averager(kld.item())

        avg_loss = train_averager()
        train_loss_history.append(avg_loss)
        
        status = f"Epoch {epoch} | Loss: {avg_loss:.2f} | BCE: {bce_averager():.1f} | Beta: {current_beta:.2f} [AMP]"
        refresh_bar(epoch_bar, status)

        # In train-loop visualizations
        if epoch % 10 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    recon_logits, mu_test, _ = model(fixed_x)
                    # Sigmoid now acts here
                    recon_test = torch.sigmoid(recon_logits)
                
                plot_latent_space(mu_test, named_labels, epoch, save_path=f"{out_dir}/latent_e{epoch}.html")
                plot_reconstructions(model, fixed_x, device, epoch, save_path=f"{out_dir}/recon_e{epoch}.png")
                save_checkpoint(model, optimizer, epoch, f"{out_dir}/checkpoint_amp.pth")

    # Final Outputs
    plot_loss(train_loss_history, title=f"{exp_name} (Mixed Precision)", save_path=f"{out_dir}/loss_history_amp.html")
    print(f"AMP Training Complete. Results in: {out_dir}")

if __name__ == "__main__":
    if sys.platform != 'win32':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
    if len(sys.argv) < 2:
        print("Usage: python train_amp.py configs/your_config.yaml")
    else:
        main(sys.argv[1])