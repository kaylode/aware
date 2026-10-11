from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ehrret.retrieval.base import Retriever
import os.path as osp
import os
import faiss

class SiameseEHREncoder(nn.Module):
    """
    Maps EHR features to an embedding space where similar patients are close.
    Trained with triplet loss or contrastive loss.
    """
    def __init__(self, n_features: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)  # L2 normalized embeddings


class TripletDataset(torch.utils.data.Dataset):
    """Efficient dataset for triplet generation"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = y
        self.labels = np.unique(y)
        self.label_to_indices = {label: np.where(y == label)[0] for label in self.labels}
        
    def __getitem__(self, index):
        anchor_label = self.y[index]
        pos_indices = self.label_to_indices[anchor_label]
        
        # Pick positive (same label)
        if len(pos_indices) > 1:
            pos_index = np.random.choice(pos_indices[pos_indices != index])
        else:
            pos_index = index
            
        # Pick negative (different label)
        other_labels = self.labels[self.labels != anchor_label]
        neg_label = np.random.choice(other_labels)
        neg_index = np.random.choice(self.label_to_indices[neg_label])
        
        return self.X[index], self.X[pos_index], self.X[neg_index]
    
    def __len__(self):
        return len(self.y)


from .utils import EarlyStopping
from sklearn.model_selection import train_test_split

class TripletLossRetriever(Retriever):
    """
    Trains embeddings using triplet loss with efficient batch processing.
    - Anchor: query patient
    - Positive: patient with same label
    - Negative: patient with different label
    """
    def __init__(
        self, 
        model: SiameseEHREncoder, 
        margin: float = 1.0, 
        pretrained_path: str = None,
        field_to_use: str = 'raw_feature',
        k: int = 32,
        distance: str = 'cosine'
    ):
        self.model = model
        self.margin = margin
        self.index = None
        self.pretrained_path = pretrained_path
        self._is_fitted = False
        self.field_to_use = field_to_use
        self.k = k
        self.distance = distance
        
        if pretrained_path and osp.isfile(pretrained_path):
            self.load_state_dict(pretrained_path)
            
    def save_state_dict(self, path: str):
        if osp.dirname(path) != '':
            os.makedirs(osp.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")
        
    def load_state_dict(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self._is_fitted = True
        print(f"Loaded model from {path}")

    def triplet_loss(self, anchor, positive, negative):
        """Triplet loss: pulls positives close, pushes negatives away"""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def fit(self, items: List[Dict], epochs=100, lr=1e-3, batch_size=256, patience=10, **kwargs):
        """Train Siamese network with triplet loss using DataLoader"""
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        X_all = np.array([item[self.field_to_use] for item in items])
        y_all = np.array([item['label'] for item in items])

        # Stratified validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.1, stratify=y_all, random_state=42
        )

        if not self._is_fitted:
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            train_dataset = TripletDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
            val_dataset = TripletDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

            early_stopping = EarlyStopping(patience=patience, mode='min')
            
            bar = tqdm(range(epochs), desc="Training Triplet Retriever")
            for epoch in bar:
                self.model.train()
                train_loss = 0
                num_batches = 0
                for anchor, pos, neg in train_loader:
                    optimizer.zero_grad()
                    anchor_emb = self.model(anchor)
                    pos_emb = self.model(pos)
                    neg_emb = self.model(neg)
                    loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1
                
                avg_train_loss = train_loss / num_batches

                # Validation
                self.model.eval()
                val_loss = 0
                num_val_batches = 0
                with torch.no_grad():
                    for anchor, pos, neg in val_loader:
                        anchor_emb = self.model(anchor)
                        pos_emb = self.model(pos)
                        neg_emb = self.model(neg)
                        loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
                        val_loss += loss.item()
                        num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                bar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}'})
                
                if early_stopping(avg_val_loss, self.model):
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            early_stopping.restore_best_model(self.model)
            self._is_fitted = True
            if self.pretrained_path:
                self.save_state_dict(self.pretrained_path)
        
        # Build FAISS index with learned embeddings using ALL training data
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_all)
            embeddings = []
            for i in range(0, X_tensor.size(0), 2048):
                batch = X_tensor[i:i+2048]
                embeddings.append(self.model(batch))
            embeddings = torch.cat(embeddings, dim=0).numpy()
        
        if self.distance == 'cosine':
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        else:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            
        self.index.add(embeddings)
        self.train_data = embeddings

        print(f"Built FAISS Index with {self.index.ntotal} vectors")
        
    def retrieve(self, query_items: List[Dict], k=None):
        """Retrieve k nearest neighbors in embedding space"""
        if k is None:
            k = self.k
            
        if not self._is_fitted:
            raise RuntimeError("Retriever not fitted.")
            
        X_query = np.array([item[self.field_to_use] for item in query_items])
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_query)
            query_emb = []
            for i in range(0, X_tensor.size(0), 2048):
                batch = X_tensor[i:i+2048]
                query_emb.append(self.model(batch))
            query_emb = torch.cat(query_emb, dim=0).numpy()
        
        if self.distance == 'cosine':
            faiss.normalize_L2(query_emb)
            
        distances, indices = self.index.search(query_emb, k)
        return indices