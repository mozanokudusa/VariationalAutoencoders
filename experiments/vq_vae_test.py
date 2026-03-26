import os
import sys
import yaml
import copy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import get_dataloaders
from src.models import get_model
from src.utils import set_seed, make_averager, save_checkpoint, plot_loss
from src.visualizations import plot_reconstructions

def vq_vae_loss(recon_logits, x, vq_loss):
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_logits.view(recon_logits.size(0), -1), 
        x.view(x.size(0), -1), 
        reduction='sum'
    )
    norm_recon_loss = recon_loss / x.size(0)
    return norm_recon_loss + vq_loss, norm_recon_loss

def train_run(config, run_name):
    """
    Executes a single training run based on the provided config.
    Returns the average loss of the final 3 epochs to determine performance.
    """
    out_dir = f"outputs/sweeps/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    train_loader, test_loader, _ = get_dataloaders(config)
    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    # Pre-fetch fixed batch for final visualization
    batch = next(iter(test_loader))
    fixed_x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

    train_loss_history = []
    num_epochs = config['training']['epochs']
    
    # Store the last few losses to average them (prevents random spikes from ruining a score)
    recent_losses = []

    epoch_bar = tqdm(range(1, num_epochs + 1), desc=f"Training {run_name}", leave=False)
    for epoch in epoch_bar:
        model.train()
        train_averager = make_averager()
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            with autocast('cuda', enabled=(device.type == 'cuda')):
                recon_logits, vq_loss, perplexity = model(batch_x)
                loss, _ = vq_vae_loss(recon_logits, batch_x, vq_loss)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_averager(loss.item())

        avg_loss = train_averager()
        train_loss_history.append(avg_loss)
        epoch_bar.set_postfix(Loss=f"{avg_loss:.1f}", Perp=f"{perplexity.item():.1f}")
        
        # Track last 3 epochs
        if epoch > num_epochs - 3:
            recent_losses.append(avg_loss)

    # ==========================================
    # ONLY output artifacts at the VERY END
    # ==========================================
    model.eval()
    with torch.no_grad():
        with autocast('cuda', enabled=(device.type == 'cuda')):
            recon_logits, _, final_perp = model(fixed_x)
            recon_test = torch.sigmoid(recon_logits)
        
        plot_reconstructions(model, fixed_x, device, num_epochs, save_path=f"{out_dir}/recon_final.png")
        save_checkpoint(model, optimizer, num_epochs, f"{out_dir}/checkpoint.pth")
    
    plot_loss(train_loss_history, title=run_name, save_path=f"{out_dir}/loss_history.html")
    
    final_score = sum(recent_losses) / len(recent_losses)
    print(f"Run [{run_name}] Complete. Final Avg Loss: {final_score:.2f} | Final Perp: {final_perp.item():.1f}")
    
    return final_score

def main(base_config_path):
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    print("\n" + "="*50)
    print("STARTING SEQUENTIAL HYPERPARAMETER SWEEP")
    print("="*50 + "\n")

    # To save time during sweeps, you might want to force fewer epochs
    base_config['training']['epochs'] = 30 
    
    current_config = copy.deepcopy(base_config)
    
    # ---------------------------------------------------------
    # Phase 1: Learning Rate Sweep
    # ---------------------------------------------------------
    print("--- PHASE 1: Learning Rate Sweep ---")
    lr_candidates = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
    best_lr, best_lr_loss = None, float('inf')
    
    for lr in lr_candidates:
        current_config['training']['learning_rate'] = lr
        run_name = f"sweep_lr_{lr}"
        loss = train_run(current_config, run_name)
        
        if loss < best_lr_loss:
            best_lr_loss = loss
            best_lr = lr
            
    print(f">>> WINNER Phase 1: Learning Rate = {best_lr} (Loss: {best_lr_loss:.2f})\n")
    current_config['training']['learning_rate'] = best_lr # Lock it in!

    # ---------------------------------------------------------
    # Phase 2: Capacity Sweep
    # ---------------------------------------------------------
    print(f"--- PHASE 2: Capacity Sweep (Using LR={best_lr}) ---")
    cap_candidates = [32, 64, 128] # Note: 128 will use a lot of VRAM!
    best_cap, best_cap_loss = None, float('inf')
    
    for cap in cap_candidates:
        current_config['model']['capacity'] = cap
        run_name = f"sweep_cap_{cap}"
        loss = train_run(current_config, run_name)
        
        if loss < best_cap_loss:
            best_cap_loss = loss
            best_cap = cap
            
    print(f">>> WINNER Phase 2: Capacity = {best_cap} (Loss: {best_cap_loss:.2f})\n")
    current_config['model']['capacity'] = best_cap # Lock it in!

    # ---------------------------------------------------------
    # Phase 3: Embedding Dimension Sweep
    # ---------------------------------------------------------
    print(f"--- PHASE 3: Embedding Dim Sweep (LR={best_lr}, Cap={best_cap}) ---")
    dim_candidates = [32, 64, 128]
    best_dim, best_dim_loss = None, float('inf')
    
    for dim in dim_candidates:
        current_config['model']['embedding_dim'] = dim
        run_name = f"sweep_embdim_{dim}"
        loss = train_run(current_config, run_name)
        
        if loss < best_dim_loss:
            best_dim_loss = loss
            best_dim = dim
            
    print(f">>> WINNER Phase 3: Embedding Dim = {best_dim} (Loss: {best_dim_loss:.2f})\n")
    
    print("="*50)
    print("SWEEP COMPLETE. BEST CONFIGURATION:")
    print(f"Learning Rate : {best_lr}")
    print(f"Capacity      : {best_cap}")
    print(f"Embedding Dim : {best_dim}")
    print("="*50)

if __name__ == "__main__":
    if sys.platform != 'win32':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    if len(sys.argv) < 2:
        print("Usage: python sweep_vq.py configs/vq_vae_cifar10.yaml")
    else:
        main(sys.argv[1])