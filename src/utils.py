import torch
import numpy as np
import random
import plotly.graph_objects as go
import os

def set_seed(seed: int = 42):
    """Ensures reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_averager():
    """Returns a function that maintains a running average."""
    count = 0
    total = 0

    def averager(new_value=None):
        nonlocal count, total
        if new_value is None:
            return total / count if count else 0.0
        count += 1
        total += new_value
        return total / count

    return averager

def save_checkpoint(model, optimizer, epoch, path):
    """Saves the model state and optimizer state to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def plot_loss(losses, title="Training Loss", save_path=None):
    """Generates an interactive Plotly chart for the loss history."""
    # Ensure losses are standard floats (handles potential tensor inputs)
    clean_losses = [l.item() if torch.is_tensor(l) else l for l in losses]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(clean_losses) + 1)),
        y=clean_losses,
        mode='lines+markers',
        name='Total Loss'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_dark",
        font=dict(family="Courier New, monospace", size=14)
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig

def refresh_bar(bar, desc):
    """Updates the tqdm progress bar description."""
    bar.set_description(desc)
    bar.refresh()