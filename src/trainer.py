import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import make_averager, refresh_bar

class VAETrainer:
    def __init__(self, model, optimizer, device, beta=1.0):
        """
        Orchestrates the training and evaluation steps for the VAE.
        
        Args:
            model: The VAE model instance.
            optimizer: The torch optimizer (e.g., Adam).
            device: 'cuda' or 'cpu'.
            beta: The weight of the KL Divergence term (B-VAE).
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.beta = beta

    def compute_loss(self, recon_x, x, mu, logvar):
        """
        Computes the VAE loss: Reconstruction Loss + KL Divergence.
        
        The KL Divergence formula used is:
        $$KLD = -0.5 \times \sum(1 + \log(\sigma^2) - \mu^2 - \sigma^2)$$
        """
        # 1. Reconstruction Loss (Binary Cross Entropy)
        # We sum over the pixels to treat the image as a single multi-variate distribution
        recon_loss = F.binary_cross_entropy(
            recon_x.view(recon_x.size(0), -1), 
            x.view(x.size(0), -1), 
            reduction='sum'
        )

        # 2. KL Divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Normalize by batch size so the learning rate doesn't need to be 
        # adjusted wildly for different batch sizes.
        return (recon_loss + (self.beta * kld)) / x.size(0)

    def train_epoch(self, train_loader):
        """Runs one full pass over the training data."""
        self.model.train()
        train_averager = make_averager()
        
        # Internal progress bar for batches
        pbar = tqdm(train_loader, desc="  -> Batch", leave=False)
        
        for batch_x, _ in pbar:
            batch_x = batch_x.to(self.device)

            # Forward pass
            recon_x, mu, logvar = self.model(batch_x)
            loss = self.compute_loss(recon_x, batch_x, mu, logvar)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            avg_loss = train_averager(loss.item())
            refresh_bar(pbar, f"  -> Loss: {avg_loss:.4f}")

        return avg_loss

    def evaluate(self, test_loader):
        """Evaluates the model on the test set without updating weights."""
        self.model.eval()
        test_averager = make_averager()
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                recon_x, mu, logvar = self.model(batch_x)
                loss = self.compute_loss(recon_x, batch_x, mu, logvar)
                test_averager(loss.item())
                
        return test_averager()