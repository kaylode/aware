import os
import json
import os.path as osp
import numpy as np
from typing import Dict, List, Tuple, Optional
from ehrpfn.datasets import (
    get_sjh_dataset, 
    get_mimic_dataset, 
    get_openml_ehr,
    get_eicu
)

"""
Purity thresholding: you can gate PFN prompting by per-query precision@k, e.g., only use retrieval if precision@k ≥ 0.4; otherwise fall back to random context.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# Example input
# ------------------------------------------------------------
# precision_dict = {
#     "Raw kNN": np.array([...]),          # precision@k for each query
#     "Feature-weighted": np.array([...]),
#     "Metric-learned": np.array([...]),
#     "ICD-embedding": np.array([...])
# }
# k = 16
# ------------------------------------------------------------

def plot_precision_histogram(precision_dict, k=16, bins=30, smooth=True):
    """
    Plot density/histogram of per-query precision@k across retrieval methods.
    
    Args:
        precision_dict: dict[str, np.ndarray]
            Mapping {method_name: per-query precision@k values (0–1)}.
        k: int
            Context size used for retrieval (for title labeling).
        bins: int
            Number of histogram bins.
        smooth: bool
            If True, overlay smooth KDE-like curves using px.density_contour.
    """
    fig = go.Figure()

    # Choose colors automatically
    palette = px.colors.qualitative.Bold

    for i, (method, values) in enumerate(precision_dict.items()):
        color = palette[i % len(palette)]
        # Histogram trace
        fig.add_trace(go.Histogram(
            x=values,
            name=method,
            opacity=0.55,
            nbinsx=bins,
            histnorm='probability density',
            marker_color=color
        ))
        # Mean line
        # mean_val = np.mean(values)
        # fig.add_vline(x=mean_val, line_width=2, line_dash="dash",
        #               line_color=color,
        #               annotation_text=f"{method} mean={mean_val:.2f}",
        #               annotation_position="top right")

    fig.update_layout(
        title=f"Distribution of Context Label Purity (Precision@{k})",
        xaxis_title=f"Per-query Precision@{k}",
        yaxis_title="Density",
        barmode='overlay',
        template='plotly_white',
        legend_title="Retrieval Method",
        width=800, height=500
    )

    fig.update_xaxes(range=[0, 1])
    fig.show()
    return fig



def _to_2d(y):
    """Coerce labels to 2D for unified handling.
       - binary/multiclass: shape (n,), becomes (n,1)
       - multilabel: keep (n, L)
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1), "single"
    return y, "multi"

def _label_match(query_label, neighbor_labels, task: str, tol: float = 0.0):
    """Return boolean matches per neighbor for a single query."""
    if task in ("binary", "multiclass"):
        # query_label, neighbor_labels are shape (1,) after _to_2d
        return (neighbor_labels[:, 0] == query_label[0])
    elif task == "multilabel":
        # Jaccard ≥ 0.5 as a simple match criterion (adjust as needed)
        inter = np.sum((neighbor_labels & query_label).astype(int), axis=1)
        union = np.sum((neighbor_labels | query_label).astype(int), axis=1)
        jacc = np.divide(inter, np.maximum(union, 1), where=(union>0))
        return (jacc >= 0.5)
    elif task == "regression":
        # match if |y_n - y_q| <= tol (e.g., tol = clinically meaningful margin)
        return (np.abs(neighbor_labels[:, 0] - query_label[0]) <= tol)
    else:
        raise ValueError(f"Unknown task: {task}")

def precision_at_k(
    neighbor_indices: np.ndarray,     # (n_queries, k)
    y_train,                          # labels for indexed points
    y_test,                           # labels for queries (aligned order)
    task: str = "binary",
    tol: float = 0.0,
) -> np.ndarray:
    """Per-query precision@k: fraction of retrieved neighbors whose label matches the query."""
    y_train2, _ = _to_2d(y_train)
    y_test2, _  = _to_2d(y_test)

    n_q, k = neighbor_indices.shape
    prec = np.zeros(n_q, dtype=float)
    for i in range(n_q):
        neigh_y = y_train2[neighbor_indices[i]]  # (k, L or 1)
        matches = _label_match(y_test2[i], neigh_y, task=task, tol=tol)
        prec[i]  = np.mean(matches.astype(float))
    return prec  # shape (n_queries,)

def contradiction_rate(
    neighbor_indices: np.ndarray,
    y_train,
    y_test,
    task: str = "binary",
    tol: float = 0.0,
) -> float:
    """Fraction of queries where the retrieved context is majority-opposed to the query label."""
    y_train2, _ = _to_2d(y_train)
    y_test2, _  = _to_2d(y_test)

    n_q, k = neighbor_indices.shape
    contradict = 0
    valid = 0
    for i in range(n_q):
        neigh_y = y_train2[neighbor_indices[i]]
        matches = _label_match(y_test2[i], neigh_y, task=task, tol=tol)
        if np.sum(matches) < (k / 2.0):
            contradict += 1
        valid += 1
    return (contradict / float(valid)) if valid > 0 else np.nan

def cohort_mix_at_k(
    neighbor_indices: np.ndarray,
    cohort_ids_train: np.ndarray,    # e.g., hospital/site/unit IDs for indexed points
) -> np.ndarray:
    """Per-query cohort entropy in neighbors (higher = more mixed cohorts)."""
    from math import log
    ent = np.zeros(neighbor_indices.shape[0], dtype=float)
    for i in range(neighbor_indices.shape[0]):
        c = cohort_ids_train[neighbor_indices[i]]
        vals, counts = np.unique(c, return_counts=True)
        p = counts / counts.sum()
        # Shannon entropy (log base e)
        ent[i] = -(p * np.log(p + 1e-12)).sum()
    return ent

def evaluate_retriever_precision(
    retriever,                         # must expose .retrieve(queries, k) -> (dists, idx)
    X_train: np.ndarray,
    X_embs_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_embs_test: np.ndarray,
    y_test: np.ndarray,
    ks: List[int],
    task: str = "binary",
    tol: float = 0.0,
    only_positive: bool = False,        # NEW ARGUMENT
    save_dir: str = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Runs retrieval for the largest k, then slices to smaller ks.
    Returns per-k: precision_mean, contradiction_rate.
    """
    kmax = max(ks)

    retriever.fit(X_train, y=y_train, X_embs_train=X_embs_train)

    if only_positive:
        positive_mask = (y_test == 1)
        X_test = X_test[positive_mask]
        y_test = y_test[positive_mask]
        X_embs_test = X_embs_test[positive_mask]

    indexes = retriever.retrieve(X_test, k=kmax, X_embs_test=X_embs_test)

    out = {}
    for k in ks:
        idx_k = indexes[:, :k]
        p_at_k = precision_at_k(idx_k, y_train, y_test, task=task, tol=tol)
        c_rate = contradiction_rate(idx_k, y_train, y_test, task=task, tol=tol)
        # Only consider non-nan for mean/std
        valid_p = p_at_k[~np.isnan(p_at_k)]
        out[k] = {
            "query_at_k": p_at_k.tolist(),   # per-query precision@k
            "precision_mean": np.mean(valid_p).item() if valid_p.size > 0 else np.nan,
            "precision_std": np.std(valid_p).item() if valid_p.size > 0 else np.nan,
            "contradiction_rate": c_rate
        }
        # print(f"k={k} | precision@k={out[k]['precision_mean']:.3f} ± {out[k]['precision_std']:.3f} | contradiction={out[k]['contradiction_rate']:.3f}")
    
    
    return out


ROOT = "/home/mpham/workspace/source/ehrpfn"
TASK = 'ICD SEPSIS'



class MIMICIVConfig:
    # Main config variables
    data_name: str = "mimicpipe"
    data_csv: str = f"{ROOT}/data/processed/mimiciv/{TASK}/all_data.csv"
    train_label_path: str = f"{ROOT}/data/processed/mimiciv/{TASK}/split/train_labels.csv"
    test_label_path: str = f"{ROOT}/data/processed/mimiciv/{TASK}/split/test_labels.csv"
    embedding_file_path: str = f"{ROOT}/data/processed/mimiciv/{TASK}/trajectories"
    feature_selection_path: str = f"{ROOT}/outputs/all/selection/mimicpipe/{TASK}"
    selection_method: str = "shap_500"
    target_column: str = f"{TASK}"

class eICUConfig:
    data_name: str = "eicu"
    data_csv: str = f"{ROOT}/data/processed/eicu/all_data.csv"
    label_path: str = f"{ROOT}/data/processed/eicu/all_labels.csv"
    train_admission_id_file: str = f"{ROOT}/data/processed/eicu/task:{TASK}/train_admission_ids.txt"
    val_admission_id_file: str = f"{ROOT}/data/processed/eicu/task:{TASK}/val_admission_ids.txt"
    test_admission_id_file: str = f"{ROOT}/data/processed/eicu/task:{TASK}/test_admission_ids.txt"
    embedding_file_path: str = f"{ROOT}/data/processed/eicu/task:{TASK}/trajectories"
    feature_selection_path: str = f"{ROOT}/outputs/all/selection/eicu/{TASK}"
    selection_method: str = "shap_500"
    target_column: str = f"{TASK}"

class SJHCPEConfig:
    data_name: str = "sjh_cpe"
    input_folder:str = f"{ROOT}/data/raw/hipe-processed-ultimate"
    output_folder:str = None
    embedding_file_path: str = f"{ROOT}/data/processed/sjh_cpe/task:{TASK}/trajectories"
    feature_selection_path: str = f"{ROOT}/outputs/all/selection/sjh_cpe/{TASK}"
    selection_method: str = "shap_500"
    target_column: str = f"{TASK}"

if __name__ == "__main__":

    from ehrpfn.utils.retrieval import WeightedRetriever, RandomRetriever, FAISS, get_retrieval_weight
    from ehrpfn.utils.retrieval import get_retriever, get_retrieval_weight

    args = MIMICIVConfig() 


    if args.data_name == 'mimicpipe' or args.data_name.startswith('mimiciv'):
        data = get_mimic_dataset(
            data_csv=args.data_csv,
            train_label_path=args.train_label_path,
            test_label_path=args.test_label_path,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            embedding_file_path=args.embedding_file_path,
        )
    elif args.data_name == 'openml_ehr':
        data = get_openml_ehr(args.data_folder)
    elif args.data_name == 'sjh_cpe':
        data = get_sjh_dataset(
            input_folder=args.input_folder,
            output_folder=None,
            target_col=args.target_column,
            patient_ids=None,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
        )
    elif args.data_name == 'eicu':
        data = get_eicu(
            task=args.target_column,
            csv_file=args.data_csv,
            label_file=args.label_path,
            train_admission_id_file=args.train_admission_id_file,
            val_admission_id_file=args.val_admission_id_file,
            test_admission_id_file=args.test_admission_id_file,
            selection_method=args.selection_method,
            feature_selection_path=args.feature_selection_path,
            embedding_file_path=args.embedding_file_path,
        )
    else:
        raise ValueError(f"Data {args.data_name} is not supported. Please choose from ['mimiciv', 'sjh_cpe']")

    save_dir = '/home/mpham/workspace/source/ehrpfn/outputs/tmp'
    ks = [128, 256, 512, 1024, 1536, 2048]

    all_metrics = {}

    for _retrieval_method in ['dbr', 'weighted_logistics', 'weighted_feattype', 'random', 'knn' 'knn-cosine']:
        feature_names = data.get("feature_names", None)

        if _retrieval_method == 'knn-cosine':
            distance = 'cosine'
        else:
            distance = 'L2'
        
        if _retrieval_method.startswith('weighted'):
            retrieval_method, weight_type = _retrieval_method.split('_')
            retrieval_weights = get_retrieval_weight(weight_type, args.data_name, TASK, feature_names=feature_names)
        else:
            retrieval_weights = None
            retrieval_method = _retrieval_method

        retriever = get_retriever(
            retrieval_method, k=2048, 
            distance=distance, weights=retrieval_weights, 
        )

        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        if _retrieval_method in ['dbr']:
            X_embs_train = data['X_embs_train']
            X_embs_test = data['X_embs_test']
        else:
            X_embs_train = None
            X_embs_test = None

        metrics = evaluate_retriever_precision(
            retriever=retriever,
            X_train=X_train,
            X_embs_train=X_embs_train,
            y_train=y_train,
            X_test=X_test,
            X_embs_test=X_embs_test,
            y_test=y_test,
            ks=ks,
            task="binary",   # or "multiclass", "multilabel", "regression"
            only_positive=True,  # Only evaluate on positive-labeled queries
            save_dir=save_dir
        )

        with open(osp.join(save_dir, f"retrieval_precision_{_retrieval_method}_{TASK}.json"), 'w') as f:
            json.dump(metrics, f, indent=3)

        all_metrics[f"{_retrieval_method}_{TASK}"] = metrics

    # Plot histogram of precision@k for k=16 'random', 
    for k in ks:
        precision_dict = {}
        for retrieval_method in ['weighted_logistics', 'weighted_feattype', 'knn']:
            metrics = all_metrics[f"{retrieval_method}_{TASK}"]
            p_at_k = np.array(metrics[k]['query_at_k'])
            precision_dict.update({f"{retrieval_method}_{k}": p_at_k})
        fig = plot_precision_histogram(precision_dict, k=k, bins=30, smooth=True)
        fig.write_image(osp.join(save_dir, f"retrieval_precision_histogram_{k}_{TASK}.pdf"), format='pdf')
