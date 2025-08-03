from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import polars as pl
from torch.utils.data import DataLoader, Dataset
from theseus import seed_everything

from ehrpfn.trajectories.cohort import DISEASE_GROUP_TO_ICDS

RARE_DISEASE_ICDS = [i for key, i in DISEASE_GROUP_TO_ICDS.items() if key in ['HCAI']]
RARE_DISEASE_ICDS = [item for sublist in RARE_DISEASE_ICDS for item in sublist]
RARE_DISEASE_ICDS.extend([i[:5] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:4] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:3] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS = list(set(RARE_DISEASE_ICDS))

seed_everything(42)

class VisitICDDataset(Dataset):
    def __init__(self, data: List, metadata: List, mapping_json: str, load_from_mapping: Dict=None):
        
        self.data = []
        self.metadata = []
        
        for i, visit in enumerate(data):
            visit = visit.strip().upper()
            visit = visit.split(",")
            self.data.append(visit)
            self.metadata.append(metadata[i])  # Store (subject_id, admission_id)

        if load_from_mapping is None:
            self.mapping_dict = json.load(open(mapping_json, 'r'))
            self.mapping_dict = {'ICD//CM//10//'+k.upper(): v for k, v in self.mapping_dict.items()}
            self.mapping_dict['[PAD]'] = len(self.mapping_dict) + 1  
            mapping_dict1 = {'ICD//CM//10//'+k.split('//')[-1][:5] if k  != '[PAD]' else k: v for k, v in self.mapping_dict.items() if k  != '[PAD]'}
            mapping_dict2 = {'ICD//CM//10//'+k.split('//')[-1][:4] if k  != '[PAD]' else k: v for k, v in self.mapping_dict.items() if k  != '[PAD]'}
            mapping_dict3 = {'ICD//CM//10//'+k.split('//')[-1][:3] if k  != '[PAD]' else k: v for k, v in self.mapping_dict.items()}
            self.mapping_dict.update(mapping_dict1)
            self.mapping_dict.update(mapping_dict2)
            self.mapping_dict.update(mapping_dict3)
        else:
            self.mapping_dict = load_from_mapping

        self.pad_token = '[PAD]'
        self.pad_token_id = self.mapping_dict['[PAD]']

        self.tokens = []
        self.valid_metadata = []

        for i, visit in enumerate(self.data):
            visit_tokens = []
            for code in visit:
                if code.startswith('ICD//PCS//10//'): continue
                if code in self.mapping_dict:
                    visit_tokens.append(self.mapping_dict[code])
                else:
                    last_code = code.split('//')[-1]
                    code = 'ICD//CM//10//' + last_code[:4]
                    if code in self.mapping_dict:
                        visit_tokens.append(self.mapping_dict[code])
                    else:
                        last_code = code.split('//')[-1]
                        code = 'ICD//CM//10//' + last_code[:3]
                        if code in self.mapping_dict:
                            visit_tokens.append(self.mapping_dict[code])
                        else:
                            continue
            if len(visit_tokens) == 0: 
                continue
            self.tokens.append(visit_tokens)
            self.valid_metadata.append(self.metadata[i])
        
        self.metadata = self.valid_metadata


    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        visit = self.tokens[idx]
        metadata = self.metadata[idx]
        return torch.LongTensor(visit), metadata

    def collate_fn(self, batch):
        max_len = max([v.size(0) for v, _ in batch])
        padded = []
        masks = []
        for v, _ in batch:
            pad_len = max_len - v.size(0)
            if pad_len > 0:
                padded.append(F.pad(v, (0, pad_len), value=self.pad_token_id))
            else:
                padded.append(v)
            masks.append(torch.cat([torch.ones(v.size(0)), torch.zeros(pad_len)]))
        return torch.stack(padded), torch.stack(masks), [metadata for _, metadata in batch]

class VisitPoolingEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_pickle, mapping_dict=None, hidden_dim=256):
        super().__init__()

        self.mapping_dict = mapping_dict
        embeds = pickle.load(open(embedding_pickle, "rb"))
        self.embed_tensor = torch.from_numpy(embeds).float()

        # If mapping_dict size is larger than embed_tensor, we need to create extra embeddings
        if self.mapping_dict is not None and len(self.mapping_dict) > embeds.shape[0]:
            extra_size = len(self.mapping_dict) - embeds.shape[0]
            extra_embeds = torch.randn(extra_size, embeds.shape[1]) * 0.01  # Small random initialization
            self.embed_tensor = torch.cat([self.embed_tensor, extra_embeds], dim=0)

        # Two-layer MLP with activation
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, mask=None, add_noise=False, noise_std=0.01):
        embed_tensor = self.embed_tensor.to(x.device)
        x = torch.clamp(x, 0, embed_tensor.shape[0] - 1)
        embeddings = F.embedding(x, embed_tensor)
        
        # Apply noise augmentation to embeddings if requested
        if add_noise:
            embeddings = embeddings + noise_std * torch.randn_like(embeddings)

        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)
            pooled = embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = embeddings.mean(dim=1)
        
        # Two-layer MLP with activation and dropout
        x = self.linear1(pooled)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        # L2 normalize the final output
        out = F.normalize(x, dim=-1)
        return out
    

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

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def train(train_data:List, train_metadata:List, test_data:List, test_metadata:List, data_name:str=None, embed_dim = 128, latent_dim = 64, num_epochs=50):
    MIMICIV_MAPPING_JSON = "embeddings/mimic4_diagnoses.json"
    EMBEDDING_PICKLE = "embeddings/glove_emb_128d_579epoch_patient0.pkl"

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
        mapping_json=MIMICIV_MAPPING_JSON
    )
    val_dataset = VisitICDDataset(
        val_set,
        val_meta_set,
        mapping_json=MIMICIV_MAPPING_JSON,
        load_from_mapping=train_dataset.mapping_dict
    )
    test_dataset = VisitICDDataset(
        test_data,
        test_metadata,
        mapping_json=MIMICIV_MAPPING_JSON,
        load_from_mapping=train_dataset.mapping_dict
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn, drop_last=True)  # Increased batch size
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisitPoolingEncoder(embed_dim, latent_dim, EMBEDDING_PICKLE, train_dataset.mapping_dict, hidden_dim=256).to(device)
    
    # Better optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Create loss function once outside the loop
    nt_xent_loss = NTXentLoss(device, 64, temperature=0.1, use_cosine_similarity=True)  # Lower temperature

    for epoch in range(num_epochs):
        # Improved noise schedule - cap the maximum noise
        noise_std = 0.005 + min((epoch / 20) * 0.01, 0.015)  # Slower increase, lower max
        
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, (x, mask, _) in enumerate(train_dataloader):
            x = x.to(device)
            mask = mask.to(device)
            
            # Apply augmentation through the model forward pass with epoch-based noise
            z_i = model(x, mask, add_noise=True, noise_std=noise_std)
            z_j = model(x, mask, add_noise=True, noise_std=noise_std)
            
            # Use the pre-created loss function
            loss = nt_xent_loss(z_i, z_j)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            # Print progress every 100 batches
            # if batch_idx % 100 == 0:
                # print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        scheduler.step()  # Update learning rate
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, mask, _ in val_dataloader:
                x = x.to(device)
                mask = mask.to(device)
                
                # For validation, use no noise or very low noise
                z_i = model(x, mask, add_noise=False)
                z_j = model(x, mask, add_noise=True, noise_std=0.005)  # Minimal noise
                
                loss = nt_xent_loss(z_i, z_j)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        # Test phase every 10 epochs
        if (epoch + 1) % 10 == 0:
            total_test_loss = 0
            with torch.no_grad():
                for x, mask, _ in test_dataloader:
                    x = x.to(device)
                    mask = mask.to(device)
                    
                    z_i = model(x, mask, add_noise=False)
                    z_j = model(x, mask, add_noise=True, noise_std=0.005)
                    
                    loss = nt_xent_loss(z_i, z_j)
                    total_test_loss += loss.item()
            
            avg_test_loss = total_test_loss / len(test_dataloader)
            print(f"Epoch {epoch+1}: Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}, Test = {avg_test_loss:.4f}, LR = {current_lr:.6f}, Noise = {noise_std:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}, LR = {current_lr:.6f}, Noise = {noise_std:.4f}")

    # ---- 5️⃣ Save and Retrieve ---- #
    torch.save(model.state_dict(), f'/home/mpham/workspace/source/tabpfn-ehr/outputs/tabpfn/data/{data_name}/process/visit_pooling_encoder.pt')
    print("Finished training and saved model for retrieval indexing.")
    
   
    
    return model

def generate_embeddings(model, dataloader, device, dataset_name="dataset"):
    """Generate embeddings for all visits in a dataset"""
    model.eval()
    all_embeddings = []
    all_metadata = []
    print(f"Generating embeddings for {dataset_name}...")
    with torch.no_grad():
        for batch_idx, (x, mask, metadata) in enumerate(dataloader):
            x = x.to(device)
            mask = mask.to(device)
            
            # Generate embeddings without noise augmentation
            embeddings = model(x, mask, add_noise=False)
            
            all_embeddings.append(embeddings.cpu().numpy())
            # Keep track of original indices for later mapping
            all_metadata.extend(metadata)
    
    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    print(f"Generated {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]} for {dataset_name}")
    return embeddings_array, all_metadata


def save_embeddings(train_data, train_metadata, test_data, test_metadata, embed_dim=128, latent_dim=64, data_name:str=None):
    """Generate and save embeddings after training"""
    MIMICIV_MAPPING_JSON = "/home/mpham/workspace/source/tabpfn-ehr/outputs/tabpfn/embeddings/mimic4_diagnoses.json"
    EMBEDDING_PICKLE = "/home/mpham/workspace/source/tabpfn-ehr/outputs/tabpfn/embeddings/glove_emb_128d_579epoch_patient0.pkl"

    train_dataset = VisitICDDataset(
        train_data,
        train_metadata,
        mapping_json=MIMICIV_MAPPING_JSON
    )
    test_dataset = VisitICDDataset(
        test_data,
        test_metadata,
        mapping_json=MIMICIV_MAPPING_JSON,
        load_from_mapping=train_dataset.mapping_dict
    )
    
    # Create dataloaders without dropping last batch to get all data
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_dataset.collate_fn, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = VisitPoolingEncoder(embed_dim, latent_dim, EMBEDDING_PICKLE, train_dataset.mapping_dict, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(f'/home/mpham/workspace/source/tabpfn-ehr/outputs/tabpfn/data/{data_name}/process/visit_pooling_encoder.pt', map_location=device))
    print("Loaded trained model successfully")

    # Generate embeddings for all datasets
    print("Generating embeddings...")
    train_embeddings, train_metadata = generate_embeddings(
        model,  train_dataloader, device, "training set"
    )
    test_embeddings, test_metadata = generate_embeddings(
        model, test_dataloader, device, "test set"
    )

    # Save embeddings with metadata
    np.savez(f'/home/mpham/workspace/source/tabpfn-ehr/outputs/tabpfn/data/{data_name}/process/trajectories_train_embeddings.npz', embeddings=train_embeddings, metadata=train_metadata)
    np.savez(f'/home/mpham/workspace/source/tabpfn-ehr/outputs/tabpfn/data/{data_name}/process/trajectories_test_embeddings.npz', embeddings=test_embeddings, metadata=test_metadata)

    return train_embeddings, test_embeddings

def preprocess_data_mimic(data_name:str):
    data = []
    metadata = []
    
    for SPLIT in ["train", "test"]:
        # Configuration
        ROOT_DIR = f"/home/mpham/workspace/source/tabpfn-ehr/data/mimiciv-icu-gbms-events/{SPLIT}"
        input_path = f"{ROOT_DIR}/09_FinalizeData"

        # Load and filter main dataframe
        df = pl.read_parquet(input_path)
        df = df.select([
            "subject_id",
            "admission_id",
            "time",
            "code",
            "numeric_value",
            "text_value"
        ])

        df_codes = df.filter(
            pl.col("code").str.starts_with("ICD//")  
        )

        if data_name == 'mimiciv_infections':
            # Filter codes not in RARE_DISEASE_ICDS
            df_codes = df_codes.filter(
                ~pl.col("code").str.split('//').list[-1].is_in(RARE_DISEASE_ICDS)
            )

        # Create process pathways with backward-looking features
        traj_df = df_codes.group_by(["subject_id", "admission_id"]).agg(
            pl.col("code").str.join(",").alias("trajectories")
        )
        traj_df = traj_df.sort(["subject_id", "admission_id"])
        traj_df = traj_df.select([
            "subject_id",
            "admission_id",
            "trajectories"
        ])

        traj_np = traj_df.to_numpy().tolist()
        traj_list = [i[-1] for i in traj_np]  # trajectories
        meta_list = [(i[0], i[1]) for i in traj_np]  # (subject_id, admission_id)

        data.append(traj_list)
        metadata.append(meta_list)
    
    return data[0], metadata[0], data[1], metadata[1]  # Return train_data, train_metadata, test_data, test_metadata



def preprocess_data_sjh(data_name):
    data = []
    metadata = []
    
    for SPLIT in ["train", "test"]:
        # Configuration
        input_path = f"/home/mpham/workspace/source/tabpfn-ehr/data/sjh-cpe-1.0/tokenized_datasets/sjh-cpe/{SPLIT}"
        if SPLIT == "train":
            input_path += "/09_FinalizeData"
        else:
            input_path += "/08_FinalizeData"

        # Load and filter main dataframe
        df = pl.read_parquet(input_path)
        df = df.select([
            "subject_id",
            "patientid",
            "time",
            "code",
            "numeric_value",
            "text_value"
        ])

        df_codes = df.filter(
            pl.col("code").str.starts_with("ICD//")  
        )

        if data_name == 'sjh_cpe':
            # Filter codes not in RARE_DISEASE_ICDS
            df_codes = df_codes.filter(
                ~pl.col("code").str.split('//').list[-1].str.starts_with('Z22')
            )

        # Create process pathways with backward-looking features
        traj_df = df_codes.group_by(["subject_id", "patientid"]).agg(
            pl.col("code").str.join(",").alias("trajectories")
        )
        traj_df = traj_df.sort(["subject_id", "patientid"])
        traj_df = traj_df.select([
            "subject_id",
            "patientid",
            "trajectories"
        ])

        traj_np = traj_df.to_numpy().tolist()
        traj_list = [i[-1] for i in traj_np]  # trajectories
        meta_list = [(i[0], i[1]) for i in traj_np]  # (subject_id, admission_id)

        data.append(traj_list)
        metadata.append(meta_list)
    
    return data[0], metadata[0], data[1], metadata[1]  # Return train_data, train_metadata, test_data, test_metadata



if __name__ == "__main__":

    # data_name = 'mimiciv_infections'
    # train_data, train_metadata, test_data, test_metadata = preprocess_data_mimic(data_name)
    
    data_name = 'sjh_cpe'
    train_data, train_metadata, test_data, test_metadata = preprocess_data_sjh(data_name)
    
    # model = train(train_data, train_metadata, test_data, test_metadata, data_name=data_name, num_epochs=30)
    
    # Generate embeddings after training
    print("\nGenerating embeddings for all visits...")
    train_val_embeddings, test_embeddings = save_embeddings(
        train_data, train_metadata, test_data, test_metadata, data_name=data_name
    )
