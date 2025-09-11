from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import os
import os.path as osp
import polars as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from ehrpfn.utils.seed import seed_everything
from ehrpfn.utils.logger import LoggerObserver
from ehrpfn.trajectories.cohort import DISEASE_GROUP_TO_ICDS
import pandas as pd


RARE_DISEASE_ICDS = [i for key, i in DISEASE_GROUP_TO_ICDS.items() if key in ['HCAI']]
RARE_DISEASE_ICDS = [item for sublist in RARE_DISEASE_ICDS for item in sublist]
RARE_DISEASE_ICDS.extend([i[:5] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:4] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:3] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS = list(set(RARE_DISEASE_ICDS))

LOGGER = LoggerObserver.getLogger('main')

seed_everything(42)

# --- In your train loop ---
# --- Global Configurable Paths ---
MIMICIV_MAPPING_JSON = "/home/mpham/workspace/source/ehrpfn/data/embeddings/mimic4_diagnoses.json"
EMBEDDING_PICKLE = "/home/mpham/workspace/source/ehrpfn/data/embeddings/glove_emb_128d_579epoch_patient0.pkl"
PROCESSED_DATA_PATH = "/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv"
PROCESSED_EICU_DATA_PATH = "/home/mpham/workspace/source/ehrpfn/data/processed_muse/eicu"

class NTXentLoss(torch.nn.Module):
    """
    Source: SimCLR paper (https://arxiv.org/abs/2002.05709)
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, z_hard=None):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        # --- Add hard negatives if provided ---
        if z_hard is not None:
            hard_neg_sim = self.similarity_function(representations, z_hard)
            negatives = torch.cat([negatives, hard_neg_sim], dim=1)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)


class VisitICDDataset(Dataset):
    def __init__(self, data: List, metadata: List, mapping_json: str, load_from_mapping: Dict=None, histories:bool=False, use_proc_icd:bool=False):

        self.data = []
        self.metadata = []
        self.histories = histories
        self.use_proc_icd = use_proc_icd
        self.pad_token = '[PAD]'
        self.mask_token = "[MASK]"
        for i, visit in enumerate(data):
            if visit is None:
                visit = [self.pad_token]
            else:
                visit = visit.strip().upper()
                if self.histories:
                    _visit = visit.split(";")
                    _visit = [v.split(',') for v in _visit]
                    visit = _visit
                else:
                    visit = visit.split(",")
            self.data.append(visit)
            self.metadata.append(metadata[i])  # Store (subject_id, admission_id)

        if load_from_mapping is None:
            self.mapping_dict = json.load(open(mapping_json, 'r'))
            self.mapping_dict = {k.upper(): v for k, v in self.mapping_dict.items()}
            self.mapping_dict['[PAD]'] = len(self.mapping_dict) + 1  
            self.mapping_dict['[MASK]'] = len(self.mapping_dict) + 1
            mapping_dict1 = {k[:5] if k not in ['[PAD]', '[MASK]'] else k: v for k, v in self.mapping_dict.items() if k not in ['[PAD]', '[MASK]']}
            mapping_dict2 = {k[:4] if k not in ['[PAD]', '[MASK]'] else k: v for k, v in self.mapping_dict.items() if k not in ['[PAD]', '[MASK]']}
            mapping_dict3 = {k[:3] if k not in ['[PAD]', '[MASK]'] else k: v for k, v in self.mapping_dict.items()}
            self.mapping_dict.update(mapping_dict1)
            self.mapping_dict.update(mapping_dict2)
            self.mapping_dict.update(mapping_dict3)
        else:
            self.mapping_dict = load_from_mapping

        self.pad_token_id = self.mapping_dict['[PAD]']
        self.mask_token_id = self.mapping_dict['[MASK]']

        self.tokens = []
        self.valid_metadata = []

        self.unknown_codes = {}

        for i, visit in enumerate(self.data):
            visit_tokens = []

            for sub_visit in visit:
                sub_visit_tokens = []
                for code in sub_visit:
                    if load_from_mapping is None:
                        if code in self.mapping_dict:
                            sub_visit_tokens.append(self.mapping_dict[code])
                        elif code[:4] in self.mapping_dict:
                            sub_visit_tokens.append(self.mapping_dict[code[:4]])
                        elif code[:3] in self.mapping_dict:
                            sub_visit_tokens.append(self.mapping_dict[code[:3]])
                        else:
                            if code not in self.unknown_codes:
                                self.unknown_codes[code] = len(self.mapping_dict)
                                self.mapping_dict[code] = self.unknown_codes[code]
                            else:
                                sub_visit_tokens.append(self.unknown_codes[code])
                    else:
                        if code in self.mapping_dict:
                            sub_visit_tokens.append(self.mapping_dict[code])
                        elif code[:4] in self.mapping_dict:
                            sub_visit_tokens.append(self.mapping_dict[code[:4]])
                        elif code[:3] in self.mapping_dict:
                            sub_visit_tokens.append(self.mapping_dict[code[:3]])
                        else:
                            sub_visit_tokens.append(self.pad_token_id)

                if len(sub_visit_tokens) == 0:
                    continue
                visit_tokens.append(sub_visit_tokens)

            self.tokens.append(visit_tokens)
            self.valid_metadata.append(self.metadata[i])
        self.metadata = self.valid_metadata

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        visit = self.tokens[idx]
        metadata = self.metadata[idx]

        flatten_visit = []
        pos_ids = []
        if self.histories: # positional encoding
            for i, v in enumerate(visit):
                flatten_visit.extend(v)
                pos_ids.extend([i] * len(v))
        
        return torch.LongTensor(flatten_visit), torch.LongTensor(pos_ids), metadata

    def collate_fn(self, batch):
        # batch: list of (visit, pos_ids, metadata)
        max_len = max([v.size(0) for v, _, _ in batch])
        padded = []
        masks = []
        pos_ids_padded = []
        for v, pos_ids, _ in batch:
            pad_len = max_len - v.size(0)
            if pad_len > 0:
                padded.append(F.pad(v, (0, pad_len), value=self.pad_token_id))
                pos_ids_padded.append(F.pad(pos_ids, (0, pad_len), value=0))
            else:
                padded.append(v)
                pos_ids_padded.append(pos_ids)
            masks.append(torch.cat([torch.ones(v.size(0)), torch.zeros(pad_len)]))
        return torch.stack(padded), torch.stack(masks), torch.stack(pos_ids_padded), [metadata for _, _, metadata in batch]

class VisitPoolingGRUEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_pickle=None, mapping_dict=None, hidden_dim=256, vocab_size=None, max_visits=50, embed_size=None):
        super().__init__()
        self.mapping_dict = mapping_dict
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_visits = max_visits
        self.embed_size = embed_size

        if embedding_pickle is not None:
            embeds = pickle.load(open(embedding_pickle, "rb"))
            self.embed_tensor = torch.from_numpy(embeds).float()
            if self.mapping_dict is not None and len(self.mapping_dict) > embeds.shape[0]:
                extra_size = len(self.mapping_dict) - embeds.shape[0]
                extra_embeds = torch.randn(extra_size, embeds.shape[1]) * 0.01
                self.embed_tensor = torch.cat([self.embed_tensor, extra_embeds], dim=0)

            self.embed_tensor = torch.nn.Parameter(self.embed_tensor, requires_grad=True)
            self.embed_size = self.embed_tensor.shape
        else:
            self.embed_size = embed_size
            self.embed_tensor = torch.randn(embed_size).float()
            self.embed_tensor = torch.nn.Parameter(self.embed_tensor, requires_grad=True)

        bidirectional = True
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        # Positional embedding for visit index
        self.pos_embedding = nn.Embedding(max_visits, input_dim)
        # --- MLM head ---
        if vocab_size is None:
            vocab_size = len(self.mapping_dict)
        if bidirectional:
            self.mlm_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, vocab_size)
            )
        else:
            self.mlm_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size)
            )

    def forward(self, x, mask=None, pos_ids=None, add_noise=False, noise_std=0.01, mlm_labels=None, mlm_mask=None):
        embed_tensor = self.embed_tensor.to(x.device)
        x = torch.clamp(x, 0, embed_tensor.shape[0] - 1)
        embeddings = F.embedding(x, embed_tensor)
        # Add positional encoding
        if pos_ids is not None:
            pos_emb = self.pos_embedding(pos_ids.to(x.device))
            embeddings = embeddings + pos_emb
        if add_noise:
            embeddings = embeddings + noise_std * torch.randn_like(embeddings)
        # Pack padded sequence for GRU
        if mask is not None:
            packed_out, h_n = self.gru(embeddings)
            out = h_n.mean(0)
        else:
            packed_out, h_n = self.gru(embeddings)
            out = h_n.mean(0)
        out = self.dropout(out)
        out = self.linear(out)
        out = F.normalize(out, dim=-1)
        # --- MLM output ---
        mlm_loss = None
        if mlm_labels is not None and mlm_mask is not None:
            mlm_logits = self.mlm_head(packed_out)
            mlm_loss = F.cross_entropy(
                mlm_logits[mlm_mask], mlm_labels[mlm_mask], reduction='mean'
            )
        return out, mlm_loss

    def save(self, save_path):
        torch.save({
            "model": self.state_dict(),
            "mapping": self.mapping_dict,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'vocab_size': self.vocab_size,
            'max_visits': self.max_visits,
            'embed_size': self.embed_size
        }, save_path)

    @classmethod
    def load_from(cls, load_path):
        ckpt = torch.load(load_path)
        model = cls(
            input_dim=ckpt['input_dim'],
            output_dim=ckpt['output_dim'],
            mapping_dict=ckpt['mapping'],
            hidden_dim=ckpt['hidden_dim'],
            vocab_size=ckpt['vocab_size'],
            max_visits=ckpt['max_visits'],
            embed_size=ckpt['embed_size'],
        )
        model.load_state_dict(ckpt['model'])
        return model


class VisitPoolingEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_pickle, mapping_dict=None, hidden_dim=256, vocab_size=None):
        super().__init__()
        self.mapping_dict = mapping_dict
        embeds = pickle.load(open(embedding_pickle, "rb"))
        self.embed_tensor = torch.from_numpy(embeds).float()
        if self.mapping_dict is not None and len(self.mapping_dict) > embeds.shape[0]:
            extra_size = len(self.mapping_dict) - embeds.shape[0]
            extra_embeds = torch.randn(extra_size, embeds.shape[1]) * 0.01
            self.embed_tensor = torch.cat([self.embed_tensor, extra_embeds], dim=0)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # --- MLM head ---
        if vocab_size is None:
            vocab_size = len(self.mapping_dict)
        self.mlm_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, vocab_size)
        )

    def forward(self, x, mask=None, add_noise=False, noise_std=0.01, mlm_labels=None, mlm_mask=None):
        embed_tensor = self.embed_tensor.to(x.device)
        x = torch.clamp(x, 0, embed_tensor.shape[0] - 1)
        embeddings = F.embedding(x, embed_tensor)
        if add_noise:
            embeddings = embeddings + noise_std * torch.randn_like(embeddings)
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)
            pooled = embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = embeddings.mean(dim=1)
        x = self.linear1(pooled)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        out = F.normalize(x, dim=-1)
        # --- MLM output ---
        mlm_loss = None
        if mlm_labels is not None and mlm_mask is not None:
            mlm_logits = self.mlm_head(out)
            mlm_loss = F.cross_entropy(
                mlm_logits[mlm_mask], mlm_labels[mlm_mask], reduction='mean'
            )
        return out, mlm_loss

# --- Helper for MLM masking ---
def mask_tokens(inputs, pad_token_id, mask_token_id, vocab_size, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability).to(labels.device)
    special_tokens_mask = (inputs == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    # 80% MASK, 10% random, 10% original
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(labels.device).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(labels.device).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long).to(labels.device)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels, masked_indices


def train(
    train_data: List,
    train_metadata: List,
    test_data: List,
    test_metadata: List,
    data_name: str = None,
    embed_dim: int = 128,
    latent_dim: int = 64,
    num_epochs: int = 50,
    use_histories: bool = False,
    use_proc_icd: bool = False,
    use_mlm: bool = False,
    mlm_weight: float = 1.0,
    use_hard_negatives: bool = False,
    mapping_json: str = MIMICIV_MAPPING_JSON,
    embedding_pickle: str = EMBEDDING_PICKLE,
    processed_data_path: str = PROCESSED_DATA_PATH,
):
    # Split train into train and validation sets
    train_ids = np.random.choice(len(train_data), int(0.8 * len(train_data)), replace=False)
    val_ids = np.setdiff1d(np.arange(len(train_data)), train_ids)
    
    train_set = [train_data[i] for i in train_ids]
    val_set = [train_data[i] for i in val_ids]
    train_meta_set = [train_metadata[i] for i in train_ids]
    val_meta_set = [train_metadata[i] for i in val_ids]
    
    train_dataset = VisitICDDataset(
        train_set, 
        train_meta_set,
        mapping_json=mapping_json,
        histories=use_histories,
        use_proc_icd=use_proc_icd
    )
    val_dataset = VisitICDDataset(
        val_set,
        val_meta_set,
        mapping_json=mapping_json,
        load_from_mapping=train_dataset.mapping_dict,
        histories=use_histories,
        use_proc_icd=use_proc_icd
    )
    test_dataset = VisitICDDataset(
        test_data,
        test_metadata,
        mapping_json=mapping_json,
        load_from_mapping=train_dataset.mapping_dict,
        histories=use_histories,
        use_proc_icd=use_proc_icd
    )

    LOGGER.text(f"Trainset shape: {len(train_dataset)}")
    LOGGER.text(f"Valset shape: {len(val_dataset)}")
    LOGGER.text(f"Testset shape: {len(test_dataset)}")

    BATCH_SIZE = 256
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=train_dataset.collate_fn, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=train_dataset.collate_fn, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisitPoolingGRUEncoder(embed_dim, latent_dim, embedding_pickle, train_dataset.mapping_dict, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    nt_xent_loss = NTXentLoss(device, BATCH_SIZE, temperature=0.15, use_cosine_similarity=True)

    for epoch in range(num_epochs):
        noise_std = 0.005 + min((epoch / 20) * 0.01, 0.015)
        model.train()
        total_train_loss = 0
        for batch_idx, (x, mask, pos_ids, _) in enumerate(train_dataloader):
            x = x.to(device)
            mask = mask.to(device)
            pos_ids = pos_ids.to(device)
            mlm_loss = None
            if use_mlm:
                x_mlm, mlm_labels, mlm_mask = mask_tokens(
                    x.clone(), pad_token_id=train_dataset.pad_token_id,
                    mask_token_id=train_dataset.mask_token_id,
                    vocab_size=len(model.mapping_dict)
                )
                z_i, mlm_loss = model(x_mlm, mask, pos_ids, add_noise=True, noise_std=noise_std, mlm_labels=mlm_labels, mlm_mask=mlm_mask)
                z_j, _ = model(x, mask, pos_ids, add_noise=True, noise_std=noise_std)
            else:
                z_i, _ = model(x, mask, pos_ids, add_noise=True, noise_std=noise_std)
                z_j, _ = model(x, mask, pos_ids, add_noise=True, noise_std=noise_std)
            if use_hard_negatives:
                negative_ids = torch.randperm(x.size(0))
                z_hard, _ = model(x[negative_ids], mask[negative_ids], pos_ids[negative_ids], add_noise=True, noise_std=noise_std)
                loss = nt_xent_loss(z_i, z_j, z_hard=z_hard)
            else:
                loss = nt_xent_loss(z_i, z_j)
            if mlm_loss is not None:
                loss = loss + mlm_weight * mlm_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, mask, pos_ids, _ in val_dataloader:
                x = x.to(device)
                mask = mask.to(device)
                pos_ids = pos_ids.to(device)
                z_i,_ = model(x, mask, pos_ids, add_noise=False)
                z_j,_ = model(x, mask, pos_ids, add_noise=True, noise_std=0.005)
                loss = nt_xent_loss(z_i, z_j)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            total_test_loss = 0
            with torch.no_grad():
                for x, mask, pos_ids, _ in test_dataloader:
                    x = x.to(device)
                    mask = mask.to(device)
                    pos_ids = pos_ids.to(device)
                    z_i,_ = model(x, mask, pos_ids, add_noise=False)
                    z_j,_ = model(x, mask, pos_ids, add_noise=True, noise_std=0.005)
                    loss = nt_xent_loss(z_i, z_j)
                    total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_dataloader)
            LOGGER.text(f"Epoch {epoch+1}: Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}, Test = {avg_test_loss:.4f}, LR = {current_lr:.6f}, Noise = {noise_std:.4f}")
        else:
            LOGGER.text(f"Epoch {epoch+1}: Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}, LR = {current_lr:.6f}, Noise = {noise_std:.4f}")

    model_name = "visit_pooling_encoder"
    if use_mlm:
        model_name += '_mlm'
    if use_hard_negatives:
        model_name += '_ohem'
    if use_proc_icd:
        model_name += '_procicd'
    save_path = osp.join(processed_data_path, data_name)
    os.makedirs(save_path, exist_ok=True)
    model.save(osp.join(save_path, f"{model_name}.pt"))
    LOGGER.text("Finished training and saved model for retrieval indexing. Saved at " + osp.join(save_path, f"{model_name}.pt"), level=LoggerObserver.SUCCESS)
    return model

def generate_embeddings(model, dataloader, device, dataset_name="dataset"):
    """Generate embeddings for all visits in a dataset"""
    model.eval()
    all_embeddings = []
    all_metadata = []
    LOGGER.text(f"Generating embeddings for {dataset_name}...")
    with torch.no_grad():
        # for x, mask, pos_ids, _ in test_dataloader:
        for batch_idx, (x, mask, pos_ids, metadata) in enumerate(dataloader):
            x = x.to(device)
            mask = mask.to(device)
            
            # Generate embeddings without noise augmentation
            embeddings, _ = model(x, mask, add_noise=False)
            
            all_embeddings.append(embeddings.cpu().numpy())
            # Keep track of original indices for later mapping
            all_metadata.extend(metadata)
    
    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    LOGGER.text(f"Generated {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]} for {dataset_name}")
    return embeddings_array, all_metadata


def save_embeddings(
    train_data,
    train_metadata,
    test_data,
    test_metadata,
    data_name: str = None,
    use_histories: int = None,
    mapping_json: str = MIMICIV_MAPPING_JSON,
    processed_data_path: str = PROCESSED_DATA_PATH,
):
    os.makedirs(f"{processed_data_path}/{data_name}/trajectories", exist_ok=True)
    train_dataset = VisitICDDataset(
        train_data,
        train_metadata,
        mapping_json=mapping_json,
        histories=True,
        use_proc_icd=True
    )
    test_dataset = VisitICDDataset(
        test_data,
        test_metadata,
        mapping_json=mapping_json,
        load_from_mapping=train_dataset.mapping_dict,
        histories = True,
        use_proc_icd = True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_dataset.collate_fn, drop_last=False)
    breakpoint()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VisitPoolingGRUEncoder.load_from(f'{processed_data_path}/visit_pooling_encoder_mlm_ohem_procicd.pt').to(device)

    LOGGER.text("Loaded trained model successfully", level=LoggerObserver.SUCCESS)
    train_embeddings, train_metadata = generate_embeddings(model, train_dataloader, device, "training set")
    test_embeddings, test_metadata = generate_embeddings(model, test_dataloader, device, "test set")
    if use_histories is not None and use_histories > 0:
        np.savez(f'{processed_data_path}/{data_name}/trajectories/train_embeddings.npz', embeddings=train_embeddings, data=train_data, metadata=train_metadata)
        np.savez(f'{processed_data_path}/{data_name}/trajectories/test_embeddings.npz', embeddings=test_embeddings, data=test_data, metadata=test_metadata)
    else:
        np.savez(f'{processed_data_path}/{data_name}/trajectories/train_embeddings.npz', embeddings=train_embeddings, data=train_data, metadata=train_metadata)
        np.savez(f'{processed_data_path}/{data_name}/trajectories/test_embeddings.npz', embeddings=test_embeddings, data=test_data, metadata=test_metadata)
    return train_embeddings, test_embeddings

def preprocess_data_mimic(
    data_name: str,
    use_histories: int = None,
    use_proc_icd: bool = False,
    processed_data_path: str = PROCESSED_DATA_PATH,
):
    data = []
    metadata = []
    for SPLIT in ["train", "test"]:
        id_df = pl.read_csv(f"{processed_data_path}/{data_name}/split/{SPLIT}_labels.csv")
        df_codes = pl.read_csv(f"{processed_data_path}/Readmission/features/preproc_diag_icu.csv.gz")
        if use_proc_icd:
            df_proc_codes = pl.read_csv(f"{processed_data_path}/Readmission/features/preproc_proc_icd_icu.csv.gz")
            df_codes = pl.concat([df_codes, df_proc_codes], how="vertical")
        traj_df = df_codes.group_by(["subject_id", "hadm_id", 'stay_id']).agg(
            pl.col("new_icd_code").str.join(",").alias("trajectories"),
        )
        traj_df = id_df.join(traj_df, on=["subject_id", "hadm_id", "stay_id"], how="left")
        traj_df = traj_df.sort(["subject_id", "hadm_id"])
        if use_histories is not None and use_histories > 0:
            traj_df = traj_df.with_columns([
                pl.col("trajectories").shift(1).over("subject_id").alias("prev_trajectories_1"),
                pl.col("trajectories").shift(2).over("subject_id").alias("prev_trajectories_2")
            ]).with_columns([
                pl.when(pl.col("prev_trajectories_1").is_null())
                .then(pl.col("trajectories"))
                .otherwise(pl.col("prev_trajectories_1") + pl.lit(",") + pl.col("trajectories"))
                .alias("trajectories_with_prev_1"),
                pl.when(pl.col("prev_trajectories_2").is_null() | pl.col("prev_trajectories_1").is_null())
                .then(
                    pl.when(pl.col("prev_trajectories_1").is_null())
                    .then(pl.col("trajectories"))
                    .otherwise(pl.col("prev_trajectories_1") + pl.lit(";") + pl.col("trajectories"))
                )
                .otherwise(
                    pl.col("prev_trajectories_2") + pl.lit(";") + 
                    pl.col("prev_trajectories_1") + pl.lit(";") + 
                    pl.col("trajectories")
                )
                .alias("trajectories_with_prev_2"),
                pl.when(pl.col("prev_trajectories_2").is_null() | pl.col("prev_trajectories_1").is_null())
                .then(
                    pl.when(pl.col("prev_trajectories_1").is_null())
                    .then(None)
                    .otherwise(pl.col("prev_trajectories_1"))
                )
                .otherwise(
                    pl.col("prev_trajectories_2") + pl.lit(";") + 
                    pl.col("prev_trajectories_1")
                )
                .alias("trajectories_with_prev_2_only_history"),
            ])
            if SPLIT == 'train':
                traj_list = traj_df.select("trajectories_with_prev_2").to_numpy().flatten().tolist()
            else:
                traj_list = traj_df.select("trajectories_with_prev_2_only_history").to_numpy().flatten().tolist()
        else:
            traj_list = traj_df.select("trajectories").to_numpy().flatten().tolist()
        meta_list = traj_df.select(["subject_id", "hadm_id", "stay_id"]).to_numpy().tolist()
        data.append(traj_list)
        metadata.append(meta_list)
    return data[0], metadata[0], data[1], metadata[1]



def preprocess_data_eicu(
    data_name: str,
    use_histories: int = None,
    use_proc_icd: bool = False,
    processed_data_path: str = PROCESSED_EICU_DATA_PATH,
):

    from ehrpfn.models.MUSE.src.utils import load_pickle
    import sys
    sys.path.append('/home/mpham/workspace/source/ehrpfn/ehrpfn/models/MUSE')

    # Convert ICD-9 to ICD-10   
    icd9toicd10 = {}
    df = pd.read_csv("/home/mpham/workspace/source/ehrpfn/preprocess/mimiciv/mimicpipe/utils/mappings/ICD9_to_ICD10_mapping.txt", delimiter='\t')
    df3 = pd.read_csv("/home/mpham/workspace/source/ehrpfn/preprocess/mimiciv/mimicpipe/utils/mappings/icd_pcs_9_to_10_mapping.csv.gz")
    mapping = {row['diagnosis_code']: row['icd10cm'] for index, row in df.iterrows()}
    mapping3 = {row['icd_9']: row['icd_10'] for index, row in df3.iterrows()}
    icd9toicd10.update(mapping)
    icd9toicd10.update(mapping3)
    all_hosp_adm_dict = load_pickle(
        "/home/mpham/workspace/source/ehrpfn/data/processed_muse/eicu/icu_stay_dict.pkl"
    )

    train_data = []
    train_metadata = []
    test_data = []
    test_metadata = []
    train_id_df = pl.read_csv(f"{processed_data_path}/task:{data_name}/train_admission_ids.txt")
    test_id_df = pl.read_csv(f"{processed_data_path}/task:{data_name}/test_admission_ids.txt")

    train_ids = train_id_df.to_numpy().flatten().tolist()
    test_ids = test_id_df.to_numpy().flatten().tolist()

    train_ids = list(map(str, train_ids))
    test_ids = list(map(str, test_ids))

    train_dfs = {
        'patient_id': [],
        'admission_id': [],
        'icu_id': [],
        'hospital_id': [],
        'trajectories': [],
    }
    test_dfs = {
        'patient_id': [],
        'admission_id': [],
        'icu_id': [],
        'hospital_id': [],
        'trajectories': [],
    }

    for icu_id in all_hosp_adm_dict.keys():
        icu_stay = all_hosp_adm_dict[icu_id]
        codes = icu_stay.diagnosis_codes

        admission_id = icu_stay.admission_id 
        patient_id = icu_stay.patient_id  # str
        hospital_id = icu_stay.hospital_id  # int

        all_codes = []
        for c in codes:
            all_codes.extend(c[-1])

        clean_codes = []
        for code in all_codes:
            plain = code.replace('.', '')
            if plain == 'nan': continue
            if code  in icd9toicd10.keys():
                clean_codes.append(icd9toicd10[code])
            elif code  in icd9toicd10.values():
                clean_codes.append(code)
            elif plain  in icd9toicd10.keys():
                clean_codes.append(icd9toicd10[plain])
            elif plain  in icd9toicd10.values():
                clean_codes.append(plain)
            else:
                clean_codes.append(code)
        
        clean_codes = list(set(clean_codes))
        if len(clean_codes) == 0:
            clean_code = 'None'
        else:
            clean_code = ','.join(clean_codes)

        if icu_id in train_ids:
            train_dfs["admission_id"].append(admission_id)
            train_dfs["patient_id"].append(patient_id)
            train_dfs["icu_id"].append(icu_id)
            train_dfs["hospital_id"].append(hospital_id)
            train_dfs["trajectories"].append(clean_code)
            train_metadata.append([patient_id, admission_id, icu_id])
        elif icu_id in test_ids:
            test_dfs["admission_id"].append(admission_id)
            test_dfs["patient_id"].append(patient_id)
            test_dfs["hospital_id"].append(hospital_id)
            test_dfs["icu_id"].append(icu_id)
            test_dfs["trajectories"].append(clean_code)
        else:
            continue


    train_df = pl.DataFrame(train_dfs)
    test_df = pl.DataFrame(test_dfs)

    def get_trajectories(traj_df, SPLIT):
        traj_df = traj_df.sort(["patient_id", "admission_id"])
        if use_histories is not None and use_histories > 0:
            traj_df = traj_df.with_columns([
                pl.col("trajectories").shift(1).over("patient_id").alias("prev_trajectories_1"),
                pl.col("trajectories").shift(2).over("patient_id").alias("prev_trajectories_2")
            ]).with_columns([
                pl.when(pl.col("prev_trajectories_1").is_null())
                .then(pl.col("trajectories"))
                .otherwise(pl.col("prev_trajectories_1") + pl.lit(",") + pl.col("trajectories"))
                .alias("trajectories_with_prev_1"),
                pl.when(pl.col("prev_trajectories_2").is_null() | pl.col("prev_trajectories_1").is_null())
                .then(
                    pl.when(pl.col("prev_trajectories_1").is_null())
                    .then(pl.col("trajectories"))
                    .otherwise(pl.col("prev_trajectories_1") + pl.lit(";") + pl.col("trajectories"))
                )
                .otherwise(
                    pl.col("prev_trajectories_2") + pl.lit(";") + 
                    pl.col("prev_trajectories_1") + pl.lit(";") + 
                    pl.col("trajectories")
                )
                .alias("trajectories_with_prev_2"),
                pl.when(pl.col("prev_trajectories_2").is_null() | pl.col("prev_trajectories_1").is_null())
                .then(
                    pl.when(pl.col("prev_trajectories_1").is_null())
                    .then(None)
                    .otherwise(pl.col("prev_trajectories_1"))
                )
                .otherwise(
                    pl.col("prev_trajectories_2") + pl.lit(";") + 
                    pl.col("prev_trajectories_1")
                )
                .alias("trajectories_with_prev_2_only_history"),
            ])
            if SPLIT == 'train':
                traj_list = traj_df.select("trajectories_with_prev_2").to_numpy().flatten().tolist()
            else:
                traj_list = traj_df.select("trajectories_with_prev_2_only_history").to_numpy().flatten().tolist()
        else:
            traj_list = traj_df.select("trajectories").to_numpy().flatten().tolist()
        meta_list = traj_df.select(["patient_id", "admission_id", "icu_id"]).to_numpy().tolist()
        return traj_list, meta_list

    train_traj_list, train_meta_list = get_trajectories(train_df, "train")
    test_traj_list, test_meta_list = get_trajectories(test_df, "test")

    return train_traj_list, train_meta_list, test_traj_list, test_meta_list

if __name__ == "__main__":
    # --- All parameters exposed here ---
    data_name = 'MORTALITY'
    embed_dim = 128
    latent_dim = 64
    num_epochs = 1024
    use_histories = 2
    use_proc_icd = True
    use_mlm = True
    mlm_weight = 0.1
    use_hard_negatives = True
    mapping_json = MIMICIV_MAPPING_JSON
    embedding_pickle = EMBEDDING_PICKLE
    processed_data_path = PROCESSED_EICU_DATA_PATH

    # train_data, train_metadata, test_data, test_metadata = preprocess_data_mimic(
    #     data_name,
    #     use_histories=use_histories,
    #     use_proc_icd=use_proc_icd,
    #     processed_data_path=processed_data_path,
    # )
    train_data, train_metadata, test_data, test_metadata = preprocess_data_eicu(
        data_name,
        use_histories=use_histories,
        use_proc_icd=use_proc_icd,
        processed_data_path=processed_data_path,
    )
    # model = train(
    #     train_data, train_metadata, test_data, test_metadata,
    #     data_name=data_name,
    #     embed_dim=embed_dim,
    #     latent_dim=latent_dim,
    #     num_epochs=num_epochs,
    #     use_histories=use_histories,
    #     use_proc_icd=use_proc_icd,
    #     use_mlm=use_mlm,
    #     mlm_weight=mlm_weight,
    #     use_hard_negatives=use_hard_negatives,
    #     mapping_json=mapping_json,
    #     embedding_pickle=embedding_pickle,
    #     processed_data_path=processed_data_path,
    # )

    save_embeddings(
        train_data,
        train_metadata,
        test_data,
        test_metadata,
        data_name=data_name,
        use_histories=use_histories,
        mapping_json=mapping_json,
        processed_data_path=processed_data_path,
    )