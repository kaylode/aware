import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Utility functions for augmentations ---

def add_gaussian_noise(x, std=0.1):
    return x + torch.randn_like(x) * std

def mixup(x, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x

def feature_shuffle(x):
    # Shuffle features independently for each sample
    x_shuffled = x.clone()
    for i in range(x.size(0)):
        idx = torch.randperm(x.size(1))
        x_shuffled[i] = x_shuffled[i, idx]
    return x_shuffled

def feature_swap(x, swap_prob=0.1):
    # Randomly swap features between samples in the batch
    x_swapped = x.clone()
    batch_size, feat_dim = x.size()
    for i in range(batch_size):
        for j in range(feat_dim):
            if np.random.rand() < swap_prob:
                swap_idx = np.random.randint(0, batch_size)
                x_swapped[i, j] = x[swap_idx, j]
    return x_swapped

# --- Contractive loss ---
def contractive_loss(recon_x, x, model, lam=1e-4):
    mse = F.mse_loss(recon_x, x)
    # Assume model.encoder is a nn.Sequential of Linear layers
    W = None
    for m in model.encoder:
        if isinstance(m, nn.Linear):
            W = m.weight
            break
    if W is not None:
        # Compute the Frobenius norm of the Jacobian
        contractive = lam * torch.sum(W ** 2)
        return mse + contractive
    else:
        return mse

# --- Robust loss (Huber) ---
def huber_loss(recon_x, x, delta=1.0):
    return F.smooth_l1_loss(recon_x, x, beta=delta)

def l1_loss(recon_x, x):
    return F.l1_loss(recon_x, x)

# --- Example experiment loop ---
def experiment(model, X, aug='gaussian', robust='mse', device='cuda'):
    model.train()
    X = torch.tensor(X, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        # --- Data augmentation ---
        if aug == 'gaussian':
            X_aug = add_gaussian_noise(X, std=0.2)
        elif aug == 'mixup':
            X_aug = mixup(X)
        elif aug == 'shuffle':
            X_aug = feature_shuffle(X)
        elif aug == 'swap':
            X_aug = feature_swap(X)
        else:
            X_aug = X

        # --- Forward ---
        recon = model(X_aug)
        # --- Loss ---
        if robust == 'huber':
            loss = huber_loss(recon, X)
        elif robust == 'l1':
            loss = l1_loss(recon, X)
        elif robust == 'contractive':
            loss = contractive_loss(recon, X, model)
        else:
            loss = F.mse_loss(recon, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# --- Usage example ---
if __name__ == "__main__":
    # Dummy model and data for demonstration
    class DummySDAE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, dim),
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    # Replace with your real data
    X = np.random.randn(512, 100).astype(np.float32)
    model = DummySDAE(dim=100).cuda()

    # Try different augmentations and losses
    print("=== Gaussian Noise + MSE ===")
    experiment(model, X, aug='gaussian', robust='mse')
    print("=== Mixup + Huber ===")
    experiment(model, X, aug='mixup', robust='huber')
    print("=== Feature Shuffle + L1 ===")
    experiment(model, X, aug='shuffle', robust='l1')
    print("=== Feature Swap + Contractive ===")
    experiment(model, X, aug='swap', robust='contractive')