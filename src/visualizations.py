import torch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from torchvision.utils import make_grid
from scipy.stats import norm
import os

def plot_latent_space(mus, labels, epoch, save_path=None):
    """
    Creates an interactive 2D scatter plot of the latent bottleneck.
    """
    mus_np = mus.detach().cpu().numpy()
    # Labels should be a list of strings (class names)
    labels_np = np.array(labels)

    fig = px.scatter(
        x=mus_np[:, 0], y=mus_np[:, 1],
        color=labels_np,
        labels={'x': 'Latent Dim 1', 'y': 'Latent Dim 2', 'color': 'Class'},
        title=f"Latent Space (Epoch {epoch})",
        template="plotly_dark",
        opacity=0.6
    )
    
    if save_path:
        fig.write_html(save_path)
    return fig

def plot_reconstructions(model, x_tensor, device, epoch, save_path, num_images=8):
    """
    Accepts a pre-fetched tensor of images and saves a comparison grid.
    Top row: Original | Bottom row: Reconstructed
    """
    model.eval()
    with torch.no_grad():
        # Ensure we don't try to plot more images than we have in the batch
        n = min(x_tensor.size(0), num_images)
        x = x_tensor[:n].to(device)
        
        # Get reconstructions from the model
        x_recon, _, _ = model(x)
        
        # Combine into a grid (Originals on top, Reconstructions on bottom)
        comparison = torch.cat([x, x_recon])
        grid = make_grid(comparison.cpu(), nrow=n, normalize=True)
        
        # Convert to numpy and plot
        plt.figure(figsize=(n * 1.5, 3))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.title(f"Reconstructions - Epoch {epoch}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_latent_manifold(model, device, epoch, save_path, n=15, img_size=32, channels=1):
    """
    Generates a 2D manifold traversal (Kingma style).
    """
    model.eval()
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    figure = np.zeros((channels, img_size * n, img_size * n))
    
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                x_decoded = model.decoder(z_sample).cpu()
                digit = x_decoded[0].reshape(channels, img_size, img_size)
                figure[:, i * img_size: (i + 1) * img_size,
                       j * img_size: (j + 1) * img_size] = digit

    plt.figure(figsize=(10, 10))
    if channels == 1:
        plt.imshow(figure[0], cmap='Greys_r')
    else:
        plt.imshow(figure.transpose(1, 2, 0))
        
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()