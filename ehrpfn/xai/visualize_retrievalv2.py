import os
import os.path as osp
import hydra
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from ehrpfn.utils import LoggerObserver
from omegaconf import DictConfig
from ehrpfn.datasets import get_mimic_dataset, get_sjh_dataset, get_mimic_dataset_v2
from ehrpfn.utils.retrieval import FAISS, ProcessRetriever, TrajectoriesRetriever
from sklearn.manifold import TSNE
import pandas as pd
import polars as pl

LOGGER = LoggerObserver.getLogger("main")

def plot_neighbors_plotly(reduced, neighbor_idxs, labels, query_idx, method_name, k=5):
    """
    Create an interactive Plotly visualization of nearest neighbors in embedding space using t-SNE.
    """
    # Create visualization dataframe
    viz_df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': labels,
    })

    # FIXED: Define colors for each label - check what type your labels are
    print(f"Label types in viz: {type(labels[0])}, unique labels: {np.unique(labels)}")
    
    # Handle both string and integer labels
    color_map = {
        0: '#636EFA',    # Blue (None)
        1: '#EF553B',    # Red (BSI)
        2: '#00CC96',    # Green (VAP)
        3: '#AB63FA'     # Purple (SEPSIS)
    }

    # Create the main scatter plot (background layer)
    fig = px.scatter(
        viz_df, 
        x='x', 
        y='y', 
        color='label',
        color_discrete_map=color_map,
        title=f'{method_name} Retrieval Space',
        width=800,
        height=600
    )


    # Update layout for better appearance
    fig.update_layout(
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2',
        legend_title='Infection Type',
        font=dict(size=12),
        template='plotly_white',
        showlegend=True
    )

    # FIXED: Update markers for better visibility - make them larger and more opaque
    fig.update_traces(
        marker=dict(size=6, opacity=0.7),  # Reduced size for background points
        selector=dict(mode='markers')
    )

    # Highlight nearest neighbors FIRST (so they appear on top of background but below query)
    if len(neighbor_idxs) > 0:
        valid_neighbor_idxs = neighbor_idxs[neighbor_idxs < len(reduced)]
        
        if len(valid_neighbor_idxs) > 0:
            fig.add_trace(go.Scatter(
                x=reduced[valid_neighbor_idxs, 0],
                y=reduced[valid_neighbor_idxs, 1],
                mode='markers',
                name=f'Top-{len(valid_neighbor_idxs)} Neighbors',
                marker=dict(
                    size=10,
                    color='rgba(255,255,255,0.9)',  # More opaque white fill
                    line=dict(color='black', width=2)  # Reduced thickness from 4 to 2
                ),
                customdata=valid_neighbor_idxs,
                showlegend=True
            ))

    # Highlight query point LAST (so it appears on top of everything)
    query_label = labels[query_idx] if query_idx < len(labels) else 'Unknown'
    fig.add_trace(go.Scatter(
        x=[reduced[query_idx, 0]],
        y=[reduced[query_idx, 1]],
        mode='markers',
        name=f'Query ({query_label})',  # Include label in legend
        marker=dict(
            size=20,  # Made larger
            color='gold',
            symbol='star',
            line=dict(color='black', width=2)  # Reduced thickness from 3 to 2
        ),
        showlegend=True  # Show in legend
    ))

    return fig


def get_neighbor_indexes(X_train, y_train, X_train_traj, X_train_process, X_test, y_test, X_test_traj, X_test_process, k=128, output_dir=None):
    # randomly select 1024 samples for visualization, but ensure classes are balanced
    np.random.seed(42)  # For reproducibility
    # Ensure we have at least 512 samples from each class if possible
    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_samples_per_class = 1024
    chosen_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        if len(cls_indices) >= min_samples_per_class:
            chosen_cls_indices = np.random.choice(cls_indices, size=min_samples_per_class, replace=False)
        else:
            chosen_cls_indices = np.random.choice(cls_indices, size=min_samples_per_class, replace=True)
        chosen_indices.extend(chosen_cls_indices)
    chosen_indices = list(set(chosen_indices))  # Remove duplicates
    chosen_indices = np.array(chosen_indices)
    # chosen_indices = np.arange(y_train.shape[0])  # Use all training samples for now

    context = X_train[chosen_indices]
    context_dbr = X_train_traj[chosen_indices]
    context_tbr = X_train_process[chosen_indices]
    labels = y_train[chosen_indices]

    # Sample k balanced classes from test set
    unique_classes, counts = np.unique(y_test, return_counts=True)
    query_idx = []
    for cls in unique_classes:
        cls_indices = np.where(y_test == cls)[0]
        if len(cls_indices) >= min_samples_per_class:
            chosen_cls_indices = np.random.choice(cls_indices, size=min_samples_per_class, replace=False)
        else:
            chosen_cls_indices = np.random.choice(cls_indices, size=min_samples_per_class, replace=True)
        query_idx.extend(chosen_cls_indices)

    query_idx = list(set(query_idx))  # Remove duplicates
    query_idx = np.array(query_idx)
    # query_idx = np.arange(y_test.shape[0])  # Use all training samples for now


    query = X_test[query_idx]  # First 1024 samples from test set
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    query_label = y_test[query_idx]  # Corresponding label for the query
    query_dbr = X_test_traj[query_idx]
    query_tbr = X_test_process[query_idx]

    return visualize_all(
        context, context_tbr, context_dbr, labels, 
        query, query_tbr, query_dbr, query_label,
        k=k, output_dir=output_dir
    )


def visualize_retrieval():
    """
    Extract top features using different feature selection methods for a single model.
    
    Args:
        args: Hydra configuration object containing all parameters
        
    Returns:
        tuple: (rankings_dict, scores_dict)
            - rankings_dict: Feature rankings
            - scores_dict: Feature importance scores
    """

    ROOT = "/home/mpham/workspace/source/ehrpfn"

    train_labels = {}
    test_labels = {}
    # infections = ['ICD SEPSIS', 'ICD UTI', 'ICD BSI', 'ICD VAP', 'ICD SSI']
    infections = ['Readmission', 'Mortality', 'Length of Stay']
    for infection in infections:

        data_csv= f"{ROOT}/data/processed/mimiciv/{infection}/all_data.csv"
        process_file_path= f"{ROOT}/data/processed/mimiciv/{infection}/process"
        trajectory_file_path= f"{ROOT}/data/processed/mimiciv/{infection}/trajectories"
        train_label_path = f"{ROOT}/data/processed/mimiciv/{infection}/split/train_labels.csv"
        test_label_path = f"{ROOT}/data/processed/mimiciv/{infection}/split/test_labels.csv"
        feature_selection_path = f"{ROOT}/outputs/tabpfn/selection/mimicpipe/{infection}"
        selection_method = "shap"
        output_dir = f"{ROOT}/outputs/tabpfn/retrieval/mimicpipe/{infection}"
        os.makedirs(output_dir, exist_ok=True)

        # Setup directories
        data = get_mimic_dataset_v2(
            data_csv=data_csv,
            train_label_path=train_label_path,
            test_label_path=test_label_path,
            feature_selection_path=feature_selection_path,
            selection_method=selection_method,
            process_file_path=process_file_path,
            trajectory_file_path=trajectory_file_path,
        )
        train_ids = data['train_ids'] #.reset_index()
        test_ids = data['test_ids'] #.reset_index()
        X_train = data['X_train']
        X_test = data['X_test']
        X_train_traj = data['X_train_traj']
        X_test_traj = data['X_test_traj']
        X_train_process = data['X_train_process']
        X_test_process = data['X_test_process']
        y_train = data['y_train']
        y_test = data['y_test']

        train_label = pd.DataFrame({
            "subject_id": train_ids["subject_id"],
            "stay_id": train_ids["stay_id"],
            "hadm_id": train_ids["hadm_id"],
            infection : y_train
        })

        test_label = pd.DataFrame({
            "subject_id": test_ids["subject_id"],
            "stay_id": test_ids["stay_id"],
            "hadm_id": test_ids["hadm_id"],
            infection : y_test
        })


        get_neighbor_indexes(
            X_train, y_train, X_train_traj, X_train_process, 
            X_test, y_test, X_test_traj, X_test_process, 
            k=128, output_dir=output_dir
        )

        train_label.to_csv(osp.join(output_dir, 'train_labels.csv'), index=False)
        test_label.to_csv(osp.join(output_dir, 'test_labels.csv'), index=False)

        # train_labels[infection] = train_label
        # test_labels[infection] = test_label

    # all_train_labels = pd.merge(
    #     train_labels['ICD BSI'],
    #     train_labels['ICD SEPSIS'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # all_train_labels = pd.merge(
    #     all_train_labels,
    #     train_labels['ICD VAP'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # all_train_labels = pd.merge(
    #     all_train_labels,
    #     train_labels['ICD UTI'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # all_train_labels = pd.merge(
    #     all_train_labels,
    #     train_labels['ICD SSI'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # def get_infections(row):
    #     return [inf for inf in infections if row[inf] == 1]
    # all_train_labels['infection'] = all_train_labels.apply(get_infections, axis=1)
    # train_exploded_labels = all_train_labels.explode('infection')
    # train_exploded_labels = train_exploded_labels[train_exploded_labels['infection'].notnull()]

    # all_test_labels = pd.merge(
    #     test_labels['ICD BSI'],
    #     test_labels['ICD SEPSIS'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # all_test_labels = pd.merge(
    #     all_test_labels,
    #     test_labels['ICD VAP'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # all_test_labels = pd.merge(
    #     all_test_labels,
    #     test_labels['ICD UTI'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # all_test_labels = pd.merge(
    #     all_test_labels,
    #     test_labels['ICD SSI'],
    #     on = ['subject_id', 'stay_id', 'hadm_id'],
    #     how='outer'
    # )
    # def get_infections(row):
    #     return [inf for inf in infections if row[inf] == 1]
    # all_test_labels['infection'] = all_test_labels.apply(get_infections, axis=1)
    # test_exploded_labels = all_test_labels.explode('infection')
    # test_exploded_labels = test_exploded_labels[test_exploded_labels['infection'].notnull()]

    # breakpoint()

    # data['train_ids'] = train_exploded_labels[['subject_id', 'stay_id', 'hadm_id']]
    # data['test_ids'] = test_exploded_labels[['subject_id', 'stay_id', 'hadm_id']]
    # data['X_train'] = data['X_train'][train_exploded_labels['index'].values]
    # data['X_train_traj'] = data['X_train_traj'][train_exploded_labels['index'].values]
    # data['X_train_process'] = data['X_train_process'][train_exploded_labels['index'].values]
    # data['X_test'] = data['X_test'][test_exploded_labels['index'].values]
    # data['X_test_traj'] = data['X_test_traj'][test_exploded_labels['index'].values]
    # data['X_test_process'] = data['X_test_process'][test_exploded_labels['index'].values]

    # y_train = train_exploded_labels['infection'].values
    # y_test = test_exploded_labels['infection'].values

def visualize_all(context, context_tbr, context_dbr, labels, query, query_tbr, query_dbr, query_label, k=32, output_dir = '/home/mpham/workspace/source/ehrpfn/ehrpfn/xai/retrievalv2'):
    # Combine labels for encoding
    all_labels_str = np.concatenate((query_label, labels))
    
    # Apply t-SNE dimensionality reduction
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    
    # Combine query and context for t-SNE
    combined_data = np.concatenate((query, context), axis=0)
    print(f"Combined data shape: {combined_data.shape}")
    
    reduced = tsne.fit_transform(combined_data)
    print(f"Reduced data shape: {reduced.shape}")


    # Initialize retrievers
    retriever = FAISS(context)
    process_retriever = ProcessRetriever(None, X_processes=context_tbr, pathway_column='pathway_with_prev_2')
    disease_retriever = TrajectoriesRetriever(None, X_traj=context_dbr)

    # Get neighbor indices
    neighbor_idxs_knn = retriever.retrieve(query, k=k)
    neighbor_idxs_process = process_retriever.retrieve(queries=None, queries_processes=query_tbr, k=k)
    neighbor_idxs_disease = disease_retriever.retrieve(queries=None, queries_traj=query_dbr, k=k)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    np.save(osp.join(output_dir, 'reduced.npy'), reduced)
    np.save(osp.join(output_dir, 'neighbor_idxs_knn.npy'), neighbor_idxs_knn)
    np.save(osp.join(output_dir, 'neighbor_idxs_process.npy'), neighbor_idxs_process)
    np.save(osp.join(output_dir, 'neighbor_idxs_disease.npy'), neighbor_idxs_disease)
    np.save(osp.join(output_dir, 'all_labels_encoded.npy'), all_labels_str)


if __name__ == "__main__":
    # For single fold feature extraction
    visualize_retrieval()

