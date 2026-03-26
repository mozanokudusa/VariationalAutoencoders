import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Import your VQ-VAE directly from where you placed it
from src.models.vq_vae import VQVAE 
from src.datasets import get_dataloaders
from src.utils import set_seed, make_averager, save_checkpoint, plot_loss, refresh_bar
from src.visualizations import plot_reconstructions

def vq_vae_loss(recon_logits, x, vq_loss):
    """
    Combines standard BCE (with logits) and the internal VQ commitment loss.
    """
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_logits.view(recon_logits.size(0), -1), 
        x.view(x.size(0), -1), 
        reduction='sum'
    )
    # Normalize recon loss by batch size and add the VQ loss
    norm_recon_loss = recon_loss / x.size(0)
    total_loss = norm_recon_loss + vq_loss
    
    return total_loss, norm_recon_loss

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_name = f"{os.path.basename(config_path).replace('.yaml', '')}_vq"
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    print(f"Starting VQ-VAE Experiment: {exp_name}")
    
    train_loader, test_loader, class_names = get_dataloaders(config)
    
    # Initialize the model directly
    model = VQVAE(
        input_channels=config['model']['input_channels'],
        hidden_channels=config['model']['capacity'],
        num_embeddings=config['model']['num_embeddings'],
        embedding_dim=config['model']['embedding_dim'],
        commitment_cost=config['model'].get('commitment_cost', 0.25)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    # Pre-fetch visualization batch
    try:
        batch = next(iter(test_loader))
        fixed_x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
    except Exception as e:
        print(f"Error during data pre-fetch: {e}")
        sys.exit(1)

    train_loss_history = []
    num_epochs = config['training']['epochs']

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        train_averager = make_averager()
        recon_averager = make_averager()
        vq_averager = make_averager()
        perp_averager = make_averager()
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            with autocast('cuda', enabled=(device.type == 'cuda')):
                # VQ-VAE returns logits, the internal quantization loss, and perplexity
                recon_logits, vq_loss, perplexity = model(batch_x)
                loss, recon_loss = vq_vae_loss(recon_logits, batch_x, vq_loss)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_averager(loss.item())
            recon_averager(recon_loss.item())
            vq_averager(vq_loss.item())
            perp_averager(perplexity.item())

        avg_loss = train_averager()
        train_loss_history.append(avg_loss)
        
        status = f"E{epoch} | Loss: {avg_loss:.1f} | Recon: {recon_averager():.1f} | VQ: {vq_averager():.3f} | Perp: {perp_averager():.1f}"
        refresh_bar(epoch_bar, status)

        if epoch % 10 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    recon_logits, _, _ = model(fixed_x)
                    # Convert logits to probabilities for visualization!
                    recon_test = torch.sigmoid(recon_logits)
                
                # Note: We skip plot_latent_space because the latent space is now discrete (a grid of vectors)
                plot_reconstructions(model, fixed_x, device, epoch, save_path=f"{out_dir}/recon_e{epoch}.png")
                save_checkpoint(model, optimizer, epoch, f"{out_dir}/checkpoint.pth")

    plot_loss(train_loss_history, title=exp_name, save_path=f"{out_dir}/loss_history.html")
    print(f"Training Complete. Artifacts in: {out_dir}")

if __name__ == "__main__":
    if sys.platform != 'win32':
        mp.set_start_method('spawn', force=True)
    if len(sys.argv) < 2:
        print("Usage: python train_vq.py configs/vq_vae_config.yaml")
    else:
        main(sys.argv[1])