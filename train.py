import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn.functional as F

# Custom module imports
from src.datasets import get_dataloaders
from src.models import get_model
from src.utils import set_seed, make_averager, save_checkpoint, plot_loss, refresh_bar
from src.visualizations import plot_latent_space, plot_reconstructions

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the VAE loss as the sum of Reconstruction Loss and KL Divergence.
    The KLD is defined as:
    """
    # Reconstruction Loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(
        recon_x.view(recon_x.size(0), -1), 
        x.view(x.size(0), -1), 
        reduction='sum'
    )
    
    # Variational Regularization (KL Divergence)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Normalized by batch size for stable gradients
    return (recon_loss + (beta * kld)) / x.size(0)

def main(config_path):
    # 1. Configuration and Output Setup
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_name = os.path.basename(config_path).replace(".yaml", "")
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting Experiment: {exp_name}")
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
    # Capturing a fixed batch here ensures worker stability and consistent visual evaluation.
    print("Pre-fetching fixed test batch...")
    try:
        test_iter = iter(test_loader)
        batch = next(test_iter)
        
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            fixed_x, fixed_y = batch[0], batch[1]
        else:
            fixed_x = batch
            fixed_y = torch.zeros(fixed_x.size(0), dtype=torch.long)
            
        fixed_x = fixed_x.to(device)
        named_labels = [class_names[i] for i in fixed_y]
    except Exception as e:
        print(f"Error during data pre-fetch: {e}")
        sys.exit(1)

    # 4. Training Loop
    train_loss_history = []
    num_epochs = config['training']['epochs']
    beta = config['model'].get('variational_beta', 1.0)

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        train_averager = make_averager()
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            recon_x, mu, logvar = model(batch_x)
            loss = vae_loss(recon_x, batch_x, mu, logvar, beta=beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_averager(loss.item())

        # Update metrics
        avg_loss = train_averager()
        train_loss_history.append(avg_loss)
        refresh_bar(epoch_bar, f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        # 5. Evaluation and Visualizations
        if epoch % 10 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                recon_test, mu_test, _ = model(fixed_x)
                
                # Save Latent Space Scatter (Interactive HTML)
                plot_latent_space(
                    mu_test, 
                    named_labels, 
                    epoch, 
                    save_path=f"{out_dir}/latent_e{epoch}.html"
                )
                
                # Save Reconstruction Grid (Static PNG)
                plot_reconstructions(
                    model, 
                    fixed_x, 
                    device, 
                    epoch, 
                    save_path=f"{out_dir}/recon_e{epoch}.png"
                )
                
                save_checkpoint(model, optimizer, epoch, f"{out_dir}/checkpoint.pth")

    # 6. Finalization
    plot_loss(train_loss_history, title=exp_name, save_path=f"{out_dir}/loss_history.html")
    print(f"Training Complete. Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    # Force 'spawn' method for multiprocessing compatibility with CUDA on Linux
    if sys.platform != 'win32':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
    if len(sys.argv) < 2:
        print("Usage: python train.py configs/your_config.yaml")
    else:
        main(sys.argv[1])