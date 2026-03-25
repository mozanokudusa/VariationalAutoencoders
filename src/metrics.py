import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

def calculate_reconstruction_error(recon_x, x):
    """Measures Mean Squared Error for pixel-perfect accuracy."""
    return F.mse_loss(recon_x, x, reduction='mean').item()

def calculate_kl_divergence(mu, logvar):
    """Measures how close the latent distribution is to a Standard Normal."""
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld.item()

def latent_silhouette_score(mus, labels):
    """
    Measures how well-clustered the different classes (digits/clothing) 
    are in the latent space. A higher score means the model learned 
    distinct features for different objects.
    """
    # Convert to numpy for sklearn
    mus_np = mus.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    if mus_np.shape[0] < 2: return 0
    return silhouette_score(mus_np, labels_np)