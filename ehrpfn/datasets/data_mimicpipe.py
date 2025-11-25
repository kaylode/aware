from typing import *
from ehrpfn.utils.logger import LoggerObserver
from ehrpfn.trajectories.cohort import DISEASE_GROUP_TO_ICDS
import os
import json
import numpy as np
import pandas as pd
import os.path as osp
from sklearn.preprocessing import LabelEncoder, StandardScaler

RARE_DISEASE_ICDS = [i for key, i in DISEASE_GROUP_TO_ICDS.items() if key in ['HCAI']]
RARE_DISEASE_ICDS = [item for sublist in RARE_DISEASE_ICDS for item in sublist]
RARE_DISEASE_ICDS.extend([i[:5] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:4] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:3] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS = list(set(RARE_DISEASE_ICDS))

LOGGER = LoggerObserver.getLogger("main")

import polars as pl


def select_features(x, y, selection_method:str, feature_selection_path:str=None, selected_indices=None):
    # Load top features
    if selection_method is not None:
        if selected_indices is None:
            if selection_method.startswith("random"):
                size = int(selection_method.split("_")[1]) if "_" in selection_method else 100
                # Random selection of features
                selected_indices = np.random.choice(x.shape[1], size=size, replace=False)
            else:
                assert feature_selection_path is not None, "feature_selection_path must be provided for feature selection"
                tmp = selection_method.split("_")
                if len(tmp) > 1:
                    selection_method, size = tmp[0], int(tmp[1])
                else:
                    size = 100
                feature_selection_path = osp.join(feature_selection_path, f"{selection_method}_feature_importance.json")
                with open(feature_selection_path, 'r') as f:
                    top_features = json.load(f)
                top_features = {k:v for k,v in sorted(top_features.items(), key=lambda item: item[1], reverse=False)}
                top_features = list(top_features.keys())[:size]
                selected_indices = [x.columns.tolist().index(col) for col in top_features if col in x.columns]
        x = x.iloc[:, selected_indices]

    feature_names = x.columns.tolist()
    return x, y, feature_names, selected_indices

def get_mimic_dataset(
        data_csv:str, 
        train_label_path:str=None, test_label_path:str=None,
        feature_selection_path:str=None, selection_method:str=None,
        embedding_file_path:str=None,
    ):

    train_ids_df = pd.read_csv(train_label_path).drop('admission_id', axis=1, errors='ignore').drop_duplicates()
    train_ids_df = train_ids_df.groupby(['subject_id', 'stay_id', 'hadm_id']).agg({
        'label': 'max'
    }).reset_index() # remove duplicates
    test_ids_df = pd.read_csv(test_label_path).drop('admission_id', axis=1, errors='ignore').drop_duplicates()
    test_ids_df = test_ids_df.groupby(['subject_id', 'stay_id', 'hadm_id']).agg({
        'label': 'max'
    }).reset_index() # remove duplicates

    df = pd.read_csv(data_csv)
    df = df.drop_duplicates(subset=['subject_id', 'stay_id', 'hadm_id'], keep='first')
    df = df.drop(columns=['label'], errors='ignore', axis=1) # use label from split file

    train_df = pd.merge(
        df,
        train_ids_df,
        left_on=["subject_id", "stay_id", 'hadm_id'],
        right_on=["subject_id", "stay_id", 'hadm_id'],
        how="inner"
    )

    test_df = pd.merge(
        df,
        test_ids_df,
        left_on=["subject_id", "stay_id", 'hadm_id'],
        right_on=["subject_id", "stay_id", 'hadm_id'],
        how="inner"
    )

    X_train_df = train_df.drop(columns=['label', 'subject_id', 'hadm_id', 'stay_id', 'edstay_id', 'admission_id', 'icustay_id'], errors='ignore')
    y_train_df = train_df['label']
    X_test_df = test_df.drop(columns=['label', 'subject_id', 'hadm_id', 'stay_id', 'edstay_id', 'admission_id', 'icustay_id'], errors='ignore')
    y_test_df = test_df['label']


    X_train_df, y_train_df, feature_names, selected_indices = select_features(
        X_train_df, y_train_df, selection_method=selection_method, 
        feature_selection_path=feature_selection_path, 
        selected_indices=None
    )

    X_test_df, y_test_df, _, _ = select_features(
        X_test_df, y_test_df, selection_method=selection_method,
        feature_selection_path=feature_selection_path,
        selected_indices=selected_indices
    )

    categorical_feature_indices = [idx for idx, i in enumerate(X_train_df.columns.tolist()) if X_train_df[i].dtype == 'object']

    label_encoder = [LabelEncoder() for _ in categorical_feature_indices]
    for idx, le in zip(categorical_feature_indices, label_encoder):
        X_train_df.iloc[:, idx] = le.fit_transform(X_train_df.iloc[:, idx])
        X_test_df.iloc[:, idx] = le.transform(X_test_df.iloc[:, idx])

    X_train = X_train_df.values
    y_train = y_train_df.values
    X_test = X_test_df.values
    y_test = y_test_df.values

    # Fit your data on the scaler object
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    LOGGER.text(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}", level=LoggerObserver.INFO)
    LOGGER.text(f"Train label distribution: {pd.Series(y_train).value_counts(normalize=True)}", level=LoggerObserver.INFO)
    LOGGER.text(f"Test label distribution: {pd.Series(y_test).value_counts(normalize=True)}", level=LoggerObserver.INFO)
    label_dist = np.bincount(np.concatenate([y_train, y_test]))

    train_ids_df = pd.DataFrame({
        "subject_id": train_df["subject_id"],
        "stay_id": train_df["stay_id"],
        "hadm_id": train_df["hadm_id"],
    })

    test_ids_df = pd.DataFrame({
        "subject_id": test_df["subject_id"],
        "stay_id": test_df["stay_id"],
        "hadm_id": test_df["hadm_id"],
    })

    if embedding_file_path is not None:
        X_train_traj = np.load(f"{embedding_file_path}/train_embeddings.npz")
        train_metadata_ids = X_train_traj['metadata'] # "subject_id", "hadm_id", "stay_id", "admission_id"
        train_embeddings = X_train_traj['embeddings']
        
        # Get embedding dimension
        embedding_dim = train_embeddings.shape[1]

        # Use left join to keep all patients from train_ids_df
        filtered_train_ids = pd.merge(
            train_ids_df,
            pd.DataFrame({
                "subject_id": train_metadata_ids[:, 0],
                "stay_id": train_metadata_ids[:, 2],
                "hadm_id": train_metadata_ids[:, 1]
            }).drop_duplicates().reset_index(),
            on=["subject_id", "stay_id", "hadm_id"],
            how="left"  # Changed from "inner" to "left"
        )

        # Create embeddings array with zeros for missing patients
        train_embedding_matrix = np.zeros((len(filtered_train_ids), embedding_dim))
        
        # Fill in embeddings for patients that have them
        valid_indices = filtered_train_ids['index'].to_numpy()
        valid_mask = ~np.isnan(valid_indices.astype(float))  # Check for non-null indices
        
        if valid_mask.any():
            train_embedding_matrix[valid_mask] = train_embeddings[valid_indices[valid_mask].astype(int)]
        
        X_train_traj = train_embedding_matrix

        # Same logic for test data
        X_test_traj = np.load(f"{embedding_file_path}/test_embeddings.npz")
        test_metadata_ids = X_test_traj['metadata']
        test_embeddings = X_test_traj['embeddings']
        
        filtered_test_ids = pd.merge(
            test_ids_df,
            pd.DataFrame({
                "subject_id": test_metadata_ids[:, 0],
                "stay_id": test_metadata_ids[:, 2],
                "hadm_id": test_metadata_ids[:, 1]
            }).drop_duplicates().reset_index(),
            on=["subject_id", "stay_id", "hadm_id"],
            how="left"  # Changed from "inner" to "left"
        )
        
        # Create embeddings array with zeros for missing patients
        test_embedding_matrix = np.zeros((len(filtered_test_ids), embedding_dim))
        
        # Fill in embeddings for patients that have them
        valid_indices = filtered_test_ids['index'].to_numpy()
        valid_mask = ~np.isnan(valid_indices.astype(float))  # Check for non-null indices
        
        if valid_mask.any():
            test_embedding_matrix[valid_mask] = test_embeddings[valid_indices[valid_mask].astype(int)]
        
        X_test_traj = test_embedding_matrix

    else:
        X_train_traj = None
        X_test_traj = None


    return {
        'X_train': X_train, 'y_train': y_train,  
        'X_test': X_test, 'y_test': y_test, 
        'X_embs_train': X_train_traj,
        'X_embs_test': X_test_traj,
        'categorical_feature_indices': categorical_feature_indices,
        'feature_names': feature_names,
        'train_ids': train_ids_df,
        'test_ids': test_ids_df,
        'class_distribution': label_dist,
    }