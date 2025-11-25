"""
retrieve.py

Experiment script for qualitative and quantitative comparison of retrieval strategies on EHR data.
Follows the step-by-step procedure outlined in steps.md.

- Compares multiple retrieval methods (kNN, weighted, embedding, etc.)
- Selects representative queries (prototypical, boundary, rare, etc.)
- Computes diagnostics (precision@k, contradiction, cohort mix, etc.)
- Produces visualizations (embedding scatter, label bar, distance hist, etc.)

Date: 2025-10-10
"""

import os
import os.path as osp
import numpy as np
import pandas as pd
import json
import umap
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

# --- Import retrieval and evaluation utilities ---
from ehrpfn.datasets import (
    get_mimic_dataset, get_eicu, get_openml_ehr, get_sjh_dataset
)
from ehrpfn.utils.retrieval import (
    WeightedRetriever, RandomRetriever, FAISS, get_retrieval_weight,
    get_retriever
)
from ehrpfn.xai.visualization.retrieval.precisionatk import (
    precision_at_k, contradiction_rate, cohort_mix_at_k, plot_precision_histogram
)


# --- Configs and dataset selection ---
ROOT = "/home/mpham/workspace/source/ehrpfn"
SAVE_DIR = osp.join(ROOT, 'outputs/tmp')
K = 2048
KS = [128, 256, 512, 1024, 1536, 2048]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class MIMICIVConfig:
    data_name = "mimicpipe"
    TASK = 'Readmission'
    data_csv = f"{ROOT}/data/processed/mimiciv/{TASK}/all_data.csv"
    train_label_path = f"{ROOT}/data/processed/mimiciv/{TASK}/split/train_labels.csv"
    test_label_path = f"{ROOT}/data/processed/mimiciv/{TASK}/split/test_labels.csv"
    embedding_file_path = f"{ROOT}/data/processed/mimiciv/{TASK}/trajectories"
    feature_selection_path = f"{ROOT}/outputs/all/selection/mimicpipe/{TASK}"
    selection_method = "shap_500"
    target_column = f"{TASK}"

class eICUConfig:
    data_name = "eicu"
    TASK = 'READMISSION'
    data_csv = f"{ROOT}/data/processed/eicu/all_data.csv"
    label_path = f"{ROOT}/data/processed/eicu/all_labels.csv"
    train_admission_id_file = f"{ROOT}/data/processed/eicu/task:{TASK}/train_admission_ids.txt"
    val_admission_id_file = f"{ROOT}/data/processed/eicu/task:{TASK}/val_admission_ids.txt"
    test_admission_id_file = f"{ROOT}/data/processed/eicu/task:{TASK}/test_admission_ids.txt"
    embedding_file_path = f"{ROOT}/data/processed/eicu/task:{TASK}/trajectories"
    feature_selection_path = f"{ROOT}/outputs/all/selection/eicu/{TASK}"
    selection_method = "shap_500"
    target_column = f"{TASK}"

class SJHCPEConfig:
    data_name = "sjh_cpe"
    TASK = 'EPISODE_CPE_INFECTION_FLAGGED'
    input_folder = f"{ROOT}/data/raw/hipe-processed-ultimate"
    output_folder = None
    embedding_file_path = f"{ROOT}/data/processed/sjh_cpe/task:{TASK}/trajectories"
    feature_selection_path = f"{ROOT}/outputs/all/selection/sjh_cpe/{TASK}"
    selection_method = "shap_500"
    target_column = f"{TASK}"

def load_data(config):
    if config.data_name == 'mimicpipe' or config.data_name.startswith('mimiciv'):
        data = get_mimic_dataset(
            data_csv=config.data_csv,
            train_label_path=config.train_label_path,
            test_label_path=config.test_label_path,
            feature_selection_path=config.feature_selection_path,
            selection_method=config.selection_method,
            embedding_file_path=config.embedding_file_path,
        )
    elif config.data_name == 'eicu':
        data = get_eicu(
            task=config.target_column,
            csv_file=config.data_csv,
            label_file=config.label_path,
            train_admission_id_file=config.train_admission_id_file,
            val_admission_id_file=config.val_admission_id_file,
            test_admission_id_file=config.test_admission_id_file,
            selection_method=config.selection_method,
            feature_selection_path=config.feature_selection_path,
            embedding_file_path=config.embedding_file_path,
        )
    elif config.data_name == 'sjh_cpe':
        data = get_sjh_dataset(
            input_folder=config.input_folder,
            output_folder=None,
            target_col=config.target_column,
            patient_ids=None,
            feature_selection_path=config.feature_selection_path,
            selection_method=config.selection_method,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data_name}")
    return data


def select_queries(X_test, y_test, n=6):
    """
    Selects representative queries:
    - prototypical positive
    - prototypical negative
    - boundary case (TODO: use model margin)
    - rare positive
    - outlier
    - subgroup (random positive)
    """
    queries = []
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    if len(pos_idx) > 0:
        queries.append(pos_idx[0])
    if len(neg_idx) > 0:
        queries.append(neg_idx[0])
    if len(pos_idx) > 1:
        queries.append(pos_idx[1])
    if len(pos_idx) > 0:
        queries.append(pos_idx[-1])
    dists = np.linalg.norm(X_test - X_test.mean(axis=0), axis=1)
    queries.append(np.argmax(dists))
    if len(pos_idx) > 2:
        queries.append(pos_idx[2])
    return queries[:n]


def plot_embedding_scatter(x_query, X_train, idx_dict, y_train, method_names, Ks, save_path_prefix=None):
    """
    UMAP 2D scatter with neighbors highlighted for each method and each K.
    - Negative label: square
    - Positive label: circle
    - Query: red star
    """
    all_idx = set()
    for idx in idx_dict.values():
        all_idx.update(idx)
    all_idx = list(all_idx)
    X_subset = np.vstack([x_query, X_train[all_idx]])
    y_subset = np.concatenate([[1], y_train[all_idx]])  # Query is always positive for marker
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=RANDOM_SEED)
    X_2d = reducer.fit_transform(X_subset)

    # Plot for each K
    for k_val in Ks:
        fig = go.Figure()
        # Plot all points (except query)
        for label, symbol in zip([0, 1], ['square', 'circle']):
            mask = (y_subset[1:] == label)
            fig.add_trace(go.Scatter(
                x=X_2d[1:,0][mask], y=X_2d[1:,1][mask],
                mode='markers',
                marker=dict(
                    color=y_subset[1:][mask],
                    colorscale='Bluered',
                    size=6,
                    symbol=symbol
                ),
                name=f'Train label {label} ({"neg" if label==0 else "pos"})',
                showlegend=True
            ))
        # Plot query
        fig.add_trace(go.Scatter(x=[X_2d[0,0]], y=[X_2d[0,1]], mode='markers',
                                 marker=dict(color='red', size=12, symbol='star'), name='Query'))
        # Overlay neighbors for each method at this K
        for i, method in enumerate(method_names):
            idx = idx_dict[method][:k_val]
            subset_idx = [all_idx.index(j) + 1 for j in idx]
            fig.add_trace(go.Scatter(x=X_2d[subset_idx,0], y=X_2d[subset_idx,1], mode='markers',
                                     marker=dict(size=10, symbol='x', line=dict(width=2, color='black')),
                                     name=f'{method} neighbors (k={k_val})', showlegend=True))
        fig.update_layout(title=f'UMAP of Query Neighborhood (k={k_val})', width=800, height=600)
        if save_path_prefix:
            fig.write_image(f"{save_path_prefix}_k{k_val}.pdf")
        fig.show()

def main():
    # --- Select dataset here ---
    only_positive = True
    DATASET = 'mimiciv'  # or 'eicu', 'sjh'
    if DATASET == 'mimiciv':
        config = MIMICIVConfig()
    elif DATASET == 'eicu':
        config = eICUConfig()
    elif DATASET == 'sjh':
        config = SJHCPEConfig()
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

    data = load_data(config)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    feature_names = data.get('feature_names', None)
    X_embs_train = data.get('X_embs_train', None)
    X_embs_test = data.get('X_embs_test', None)

    

    # --- Step 2: Build retrieval indices for each method ---
    retrieval_methods = [
        'dbr',
        'weighted_logistics',
        'weighted_feattype',
        'knn',
        'knn-cosine',
    ]
    retrievers = {}
    for method in retrieval_methods:
        if method == 'knn-cosine':
            distance = 'cosine'
        else:
            distance = 'L2'
        if method.startswith('weighted'):
            _, weight_type = method.split('_')
            weights = get_retrieval_weight(weight_type, config.data_name, config.TASK, feature_names=feature_names)
            retriever_name = 'weighted'
        else:
            weights = None
            retriever_name = method
            if method == 'knn-cosine':
                retriever_name = 'knn'
        retrievers[method] = get_retriever(
            retriever_name, k=K, distance=distance, weights=weights,
            return_distances=True
        )

        retrievers[method].fit(X_train, y=y_train, X_embs_train=X_embs_train)

    # --- Step 3: Select representative queries ---
    if only_positive:
        positive_mask = (y_test == 1)
        query_indices = np.where(positive_mask)[0].tolist()
    else:
        query_indices = select_queries(X_test, y_test, n=6)

    X_test = X_test[query_indices]
    y_test = y_test[query_indices]
    X_embs_test = X_embs_test[query_indices]
    # --- Step 4: For each query and method, retrieve neighbors and compute diagnostics ---
    # results = {}
    # for qid in query_indices:
    #     qres = {}
    #     x_query = X_test[qid]
    #     y_query = y_test[None, qid]
    #     x_emb_query = X_embs_test[qid] if X_embs_test is not None else None
    #     for method, retriever in retrievers.items():
    #         dists, idx = retriever.retrieve(x_query[None, :], X_embs_test=x_emb_query[None, :])
    #         idx = idx[0]
    #         dists = dists[0]
    #         neighbor_labels = y_train[idx]
    #         p_at_k = precision_at_k(idx[None, :], y_train, y_query, task="binary")
    #         c_rate = contradiction_rate(idx[None, :], y_train, y_query, task="binary")
    #         mean_dist = np.mean(dists)
    #         median_dist = np.median(dists)
    #         qres[method] = {
    #             'neighbor_idx': idx.tolist(),
    #             'neighbor_labels': neighbor_labels.tolist(),
    #             'distances': dists.tolist(),
    #             'precision_at_k': float(p_at_k[0]),
    #             'contradiction_rate': float(c_rate),
    #             'mean_distance': float(mean_dist),
    #             'median_distance': float(median_dist),
    #         }
    #     results[qid] = qres

    # # --- Step 5: Save results ---
    # os.makedirs(SAVE_DIR, exist_ok=True)
    # with open(osp.join(SAVE_DIR, f'retrieval_case_results_{config.TASK}.json'), 'w') as f:
    #     json.dump(results, f, indent=2)

    # # --- Step 6: Visualizations (sketch, not full implementation) ---
    # qid = query_indices[0]
    # x_query = X_test[qid]
    # idx_dict = {m: results[qid][m]['neighbor_idx'] for m in retrieval_methods}
    # plot_embedding_scatter(
    #     x_query, X_train, idx_dict, y_train, retrieval_methods, Ks=KS,
    #     save_path_prefix=osp.join(SAVE_DIR, f'embedding_scatter_{qid}')
    # )


    # --- Step 7: Global context purity histogram for each K ---
    all_precision_dict = {}
    for method, retriever in retrievers.items():
        indexes = retriever.retrieve(X_test, X_embs_test=X_embs_test)[1]

        out = {}
        for k in KS:
            idx_k = indexes[:, :k]
            p_at_k = precision_at_k(idx_k, y_train, y_test, task="binary")
            c_rate = contradiction_rate(idx_k, y_train, y_test, task="binary")
            # Only consider non-nan for mean/std
            valid_p = p_at_k[~np.isnan(p_at_k)]
            out[k] = {
                "query_at_k": p_at_k.tolist(),   # per-query precision@k
                "precision_mean": np.mean(valid_p).item() if valid_p.size > 0 else np.nan,
                "precision_std": np.std(valid_p).item() if valid_p.size > 0 else np.nan,
                "contradiction_rate": c_rate
            }

        all_precision_dict[method] = out

    for k in KS:
        k_precision_dict = {method: all_precision_dict[method][k]['query_at_k'] for method in retrieval_methods}
        fig = plot_precision_histogram(k_precision_dict, k=k, bins=30, smooth=True)
        fig.write_image(osp.join(SAVE_DIR, f'precision_histogram_{config.TASK}_k{k}.pdf'), format='pdf')

    print(f"Experiment complete for dataset {config.data_name}. Results and figures saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
