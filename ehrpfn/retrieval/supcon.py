from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import os
import os.path as osp
from tqdm import tqdm
from ehrret.retrieval.base import Retriever



class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: (B, D) normalized
        labels: (B,)
        """
        device = embeddings.device
        B = embeddings.shape[0]
        
        # Cosine similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)
        sim_matrix = sim_matrix / self.temperature

        # Mask out self-comparisons
        logits_mask = torch.ones_like(sim_matrix) - torch.eye(B, device=device)

        exp_sim = torch.exp(sim_matrix) * logits_mask

        # Label mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask * logits_mask  # remove self

        # Compute log prob
        # log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        
        # Numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Only positives
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss


class SupConEncoder(nn.Module):
    """
    Combines supervised contrastive learning + classification.
    - SupCon loss: learns good embeddings for retrieval
    - Classification head: maintains task performance
    """
    def __init__(self, n_features: int, hidden_dim:int =512, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        self.attention = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
            nn.Sigmoid()
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(n_features, 1) # Changed to use weighted_x directly
    
    def forward(self, x, return_projection=False):
        """
        Args:
            x: (batch, n_features)
            return_projection: if True, return projection for contrastive loss
        Returns:
            embeddings: (batch, embedding_dim) - learned embeddings
            proj/logits: depending on return_projection flag
        """
        weights = self.attention(x)
        weighted_x = x * weights
        
        if return_projection:
            # For contrastive loss training
            proj = F.normalize(self.projector(weighted_x), p=2, dim=1)
            return weighted_x, proj
        else:
            # For classification
            logits = self.classifier(weighted_x)
            return weighted_x, logits
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract learned embeddings for dataset"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            # Process in chunks if too large
            if X_tensor.size(0) > 2048:
                all_emb = []
                for i in range(0, X_tensor.size(0), 2048):
                    batch = X_tensor[i:i+2048]
                    weighted_x, _ = self.forward(batch, return_projection=False)
                    all_emb.append(weighted_x)
                return torch.cat(all_emb, dim=0).numpy()
            
            embeddings, _ = self.forward(X_tensor, return_projection=False)
            return embeddings.numpy()


from .utils import EarlyStopping
from sklearn.model_selection import train_test_split

class SupConRetriever(Retriever):
    """
    Supervised Contrastive Retriever with FAISS.
    End-to-end trainable with hybrid loss (SupCon + Classification).
    """
    def __init__(
        self, 
        model: SupConEncoder, 
        field_to_use: str = 'raw_feature',
        distance: str = 'cosine', 
        k: int = 32, 
        pretrained_path: str = None
    ):
        self.model = model
        self.distance = distance
        self.index = None
        self._is_fitted = False
        self.k = k
        self.field_to_use = field_to_use
        self.pretrained_path = pretrained_path
        self.return_distances = False
        
        if pretrained_path is not None and osp.isfile(pretrained_path):
            self.load_state_dict(pretrained_path)

    def save_state_dict(self, path: str):
        """Save model weights to file"""
        if osp.dirname(path) != '':
            os.makedirs(osp.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model weights to {path}")
        
    def load_state_dict(self, path: str):
        """Load pretrained model weights"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self._is_fitted = True
        print(f"Loaded model weights from {path}")
        
    def fit(self, items: List[Dict], epochs: int = 100, lr: float = 1e-3, batch_size: int = 1024, alpha: float = 0.5, patience: int = 10, **kwargs):
        """
        Train with combined loss:
        Total Loss = α * SupCon Loss + (1-α) * Classification Loss
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        X_all = np.array([item[self.field_to_use] for item in items])
        y_all = np.array([item['label'] for item in items])
        
        # Stratified validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.1, stratify=y_all, random_state=42
        )

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        if not self._is_fitted:
            import torch.optim as optim
            
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            supcon_criterion = SupConLoss(temperature=0.07)
            cls_criterion = nn.BCEWithLogitsLoss()
            
            train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False, drop_last=False)

            early_stopping = EarlyStopping(patience=patience, mode='min')
            
            bar = tqdm(range(epochs), desc="Training SupCon Retriever")
            for epoch in bar:
                self.model.train()
                epoch_supcon = 0
                epoch_cls = 0
                num_batches = 0
                
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    weighted_x, projections = self.model(xb, return_projection=True)
                    logits = self.model.classifier(weighted_x)
                    supcon_loss = supcon_criterion(projections, yb.long())
                    cls_loss = cls_criterion(logits.view(-1), yb.view(-1))
                    total_loss = alpha * supcon_loss + (1 - alpha) * cls_loss
                    total_loss.backward()
                    optimizer.step()
                    epoch_supcon += supcon_loss.item()
                    epoch_cls += cls_loss.item()
                    num_batches += 1
                
                avg_train_loss = (alpha * epoch_supcon + (1-alpha) * epoch_cls) / num_batches

                # Validation
                self.model.eval()
                val_supcon = 0
                val_cls = 0
                num_val_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        weighted_x, projections = self.model(xb, return_projection=True)
                        logits = self.model.classifier(weighted_x)
                        supcon_loss = supcon_criterion(projections, yb.long())
                        cls_loss = cls_criterion(logits.view(-1), yb.view(-1))
                        val_supcon += supcon_loss.item()
                        val_cls += cls_loss.item()
                        num_val_batches += 1
                
                avg_val_loss = (alpha * val_supcon + (1-alpha) * val_cls) / num_val_batches
                
                bar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}'
                })

                if early_stopping(avg_val_loss, self.model):
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            early_stopping.restore_best_model(self.model)
            self._is_fitted = True
            if self.pretrained_path is not None:
                self.save_state_dict(self.pretrained_path)
        
        # Build FAISS index with learned embeddings using ALL data
        self.model.eval()
        with torch.no_grad():
            X_all_tensor = torch.FloatTensor(X_all)
            embeddings = []
            for i in range(0, X_all_tensor.size(0), 2048):
                batch = X_all_tensor[i:i+2048]
                weighted_x, _ = self.model(batch, return_projection=False)
                embeddings.append(weighted_x)
            embeddings = torch.cat(embeddings, dim=0).numpy()
            embeddings = np.ascontiguousarray(embeddings)
        
        if self.distance == 'cosine':
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        else:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings)
        self.train_data = embeddings
        print(f"Built FAISS Index ({self.distance}) with {self.index.ntotal} vectors")
    
    def retrieve(self, query_items: List[Dict], k=None):
        """Retrieve k nearest neighbors using learned embeddings"""
        if k is None:
            k = self.k
            
        X_query = np.array([item[self.field_to_use] for item in query_items])
        
        if not self._is_fitted:
            raise RuntimeError("Retriever is not fitted yet.")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_query)
            query_embeddings = []
            for i in range(0, X_tensor.size(0), 2048):
                batch = X_tensor[i:i+2048]
                weighted_x, _ = self.model(batch, return_projection=False)
                query_embeddings.append(weighted_x)
            query_embeddings = torch.cat(query_embeddings, dim=0).numpy()
            query_embeddings = np.ascontiguousarray(query_embeddings)
        
        if self.distance == 'cosine':
            faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, k)
        return indices


if __name__ == "__main__":
    from ehrmeds import (
        get_mimic_dataset, get_mimiciv_default_configs,
        get_eicu, get_eicu_default_configs,
        get_sjh_dataset, get_sjh_cpe_default_configs
    )

    dataset_name = "eicu"  # "mimiciv", 'sjh_cpe'
    TASK = "MORTALITY"  # Example task

    if dataset_name == "mimiciv":
        configs = get_mimiciv_default_configs(TASK=TASK)
        dataset = get_mimic_dataset(**configs)
    elif dataset_name == "eicu":
        configs = get_eicu_default_configs(TASK=TASK)
        dataset = get_eicu(**configs)
    elif dataset_name == "sjh_cpe":
        configs = get_sjh_cpe_default_configs(TASK=TASK)
        dataset = get_sjh_dataset(**configs)

    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    # Prepare items format
    train_items = [
        {'raw_feature': X_train[i], 'label': int(y_train[i])}
        for i in range(len(X_train))
    ]
    test_items = [
        {'raw_feature': X_test[i], 'label': int(y_test[i])}
        for i in range(len(X_test))
    ]
    
    # Example usage
    n_features = X_train.shape[1]
    print(f"Number of features: {n_features}")
    
    model = SupConEncoder(
        n_features=n_features, 
        embedding_dim=128,
        dropout=0.3
    )
    
    save_dir = f"/home/mpham/workspace/source/ehrfm/ehr-retrieval/outputs/ehr-retrieval/supcon_retriever/{dataset_name}/{TASK}/"
    pretrained_path = os.path.join(save_dir, "supcon_retriever.pth")
    
    retriever = SupConRetriever(
        model=model, 
        distance='cosine',
        k=32,
        pretrained_path=pretrained_path
    )
    
    # Train with hybrid loss
    retriever.fit(train_items, epochs=100, lr=1e-3, alpha=0.5)
    
    # Retrieve neighbors
    indices = retriever.retrieve(test_items[:5])
    print("Retrieved indices shape:", indices.shape)
    print("Sample retrieved indices:", indices[:3])

