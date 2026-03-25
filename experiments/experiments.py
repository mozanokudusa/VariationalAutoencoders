import os
import sys
import yaml
import torch

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_model
from src.datasets import get_dataloaders
from src.visualizations import plot_latent_space, plot_reconstructions, plot_latent_manifold

def run_experiment(config_path, checkpoint_path):
    """
    Loads a trained model and generates all visualizations.
    """
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.dirname(__file__) # This experiments/ folder

    # 2. Load Model & Data
    train_loader, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Check if the checkpoint was saved as a full dict or just weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
    else:
        model.load_state_dict(checkpoint)
        epoch = "Final"

    model.eval()
    print(f"Model loaded from {checkpoint_path}. Generating visualizations...")

    with torch.no_grad():
        # Get a batch for the scatter plot
        test_x, test_y = next(iter(test_loader))
        test_x = test_x.to(device)
        _, mu, _ = model(test_x)

        # 3. Generate Visualizations
        print("-> Generating Latent Space Scatter (Plotly)...")
        plot_latent_space(
            mu, test_y, epoch, 
            save_path=os.path.join(exp_dir, f"latent_viz.html")
        )

        print("-> Generating Reconstruction Comparison...")
        plot_reconstructions(
            model, test_loader, device, epoch, 
            out_dir=exp_dir
        )

        print("Generating Final High-Res Visualizations...")

        # Generate a high-resolution manifold traversal
        plot_latent_manifold(
            model, device, "Final", 
            save_path="experiments/manifold_final_grid.png",
            n=20, # Higher density for the final report
            img_size=config['model']['image_size'],
            channels=config['model']['input_channels']
        )

        # 4. Manifold Traversal (Only if 2D latent space)
        if config['model']['latent_dims'] == 2:
            print("-> Generating Latent Manifold Grid...")
            plot_latent_manifold(
                model, device, epoch, 
                out_dir=exp_dir,
                img_size=config['model']['image_size'],
                channels=config['model']['input_channels']
            )

    print(f"All visualizations saved in {exp_dir}")

if __name__ == "__main__":
    # Example usage: 
    # python experiments/experiments.py configs/vae_mnist_2layer.yaml outputs/vae_mnist_2layer/checkpoint.pth
    if len(sys.argv) < 3:
        print("Usage: python experiments/experiments.py <config_path> <checkpoint_path>")
    else:
        run_experiment(sys.argv[1], sys.argv[2])