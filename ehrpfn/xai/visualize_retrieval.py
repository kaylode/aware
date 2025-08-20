import os
import os.path as osp
import hydra
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from ehrpfn.utils import LoggerObserver
from omegaconf import DictConfig
from ehrpfn.datasets import get_mimic_dataset, get_sjh_dataset
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

@hydra.main(version_base=None, config_path="../../configs", config_name="mimiciv")
def visualize_retrieval(args: DictConfig):
    """
    Extract top features using different feature selection methods for a single model.
    
    Args:
        args: Hydra configuration object containing all parameters
        
    Returns:
        tuple: (rankings_dict, scores_dict)
            - rankings_dict: Feature rankings
            - scores_dict: Feature importance scores
    """

    
    # Setup directories
    data_output_folder = osp.join(args.data_folder, args.data_name)

    if args.data_name.startswith('mimiciv'):
        data = get_mimic_dataset(
            train_folder=None,
            test_folder=None,
            icu_ids=args.icu_ids,
            data_output_folder=data_output_folder,
            target_column=None,
            train_label_path=None,
            test_label_path=None,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            process_file_path=args.process_file_path,
            n_process_variants=args.n_process_variants,
            trajectory_file_path=args.trajectory_file_path,
            logger=LOGGER
        )
    elif args.data_name == 'sjh_cpe':
        data = get_sjh_dataset(
            input_folder=args.input_folder,
            output_folder=data_output_folder,
            target_col=args.target_column,
            max_length=args.max_length,
            num_chunks=args.num_chunks,
            patient_ids=None,

            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            process_file_path=args.process_file_path,
            n_process_variants=args.n_process_variants,
            trajectory_file_path=args.trajectory_file_path,

        )
    else:
        raise ValueError(f"Data {args.data_name} is not supported. Please choose from ['mimiciv', 'sjh_cpe']")

    train_ids = data['train_ids'].with_row_index()
    test_ids = data['test_ids'].with_row_index()

    # test_ids = pd.merge(
    #     test_id_df_pandas,
    #     data['test_ids'].with_row_index().to_pandas(),
    #     on=["subject_id", "admission_id"],
    #     how="inner"
    # )

    test_label_BSI = pl.read_csv("/home/mpham/workspace/source/ehrpfn/data/labels/PATIENT_INFECTION/BSI_test_labels.csv")
    test_label_VAP = pl.read_csv("/home/mpham/workspace/source/ehrpfn/data/labels/PATIENT_INFECTION/VAP_test_labels.csv")
    test_label_SEPSIS = pl.read_csv("/home/mpham/workspace/source/ehrpfn/data/labels/PATIENT_INFECTION/SEPSIS_test_labels.csv")

    train_label_BSI = pl.read_csv("/home/mpham/workspace/source/ehrpfn/data/labels/PATIENT_INFECTION/BSI_train_labels.csv")
    train_label_VAP = pl.read_csv("/home/mpham/workspace/source/ehrpfn/data/labels/PATIENT_INFECTION/VAP_train_labels.csv")
    train_label_SEPSIS = pl.read_csv("/home/mpham/workspace/source/ehrpfn/data/labels/PATIENT_INFECTION/SEPSIS_train_labels.csv")

    
    train_id_df = train_ids.join(
        train_label_BSI, on=["subject_id", "admission_id"], how="left"
    ).join(
        train_label_VAP, on=["subject_id", "admission_id"], how="left"
    ).join(
        train_label_SEPSIS, on=["subject_id", "admission_id"], how="left"
    )
    train_id_df = train_id_df.unique()

    test_id_df = test_ids.join(
        test_label_BSI, on=["subject_id", "admission_id"], how="left"
    ).join(
        test_label_VAP, on=["subject_id", "admission_id"], how="left"
    ).join(
        test_label_SEPSIS, on=["subject_id", "admission_id"], how="left"
    )
    test_id_df = test_id_df.unique()

    # Create label column based on infection types
    def assign_label(row):
        if row['BSI'] == 1:
            return 'BSI'
        elif row['VAP'] == 1:
            return 'VAP'
        elif row['SEPSIS'] == 1:
            return 'SEPSIS'
        else:
            return 'None'

    # Convert to pandas for easier manipulation
    train_id_df_pandas = train_id_df.to_pandas()
    train_id_df_pandas['label'] = train_id_df_pandas.apply(assign_label, axis=1)
    test_id_df_pandas = test_id_df.to_pandas()
    test_id_df_pandas['label'] = test_id_df_pandas.apply(assign_label, axis=1)

    train_ids = train_id_df_pandas
    test_ids = test_id_df_pandas

    data['train_ids'] = train_ids
    data['test_ids'] = test_ids
    data['X_train'] = data['X_train'][train_ids['index'].values]
    data['X_train_traj'] = data['X_train_traj'][train_ids['index'].values]
    data['X_train_process'] = data['X_train_process'][train_ids['index'].values]
    data['X_test'] = data['X_test'][test_ids['index'].values]
    data['X_test_traj'] = data['X_test_traj'][test_ids['index'].values]
    data['X_test_process'] = data['X_test_process'][test_ids['index'].values]

    y_train = train_ids['label'].values
    y_test = test_ids['label'].values


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

    context = data['X_train'][chosen_indices]
    context_dbr = data['X_train_traj'][chosen_indices]
    context_tbr = data['X_train_process'][chosen_indices]
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


    query = data['X_test'][query_idx]  # First 1024 samples from test set
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    query_label = y_test[query_idx]  # Corresponding label for the query
    query_dbr = data['X_test_traj'][query_idx]
    query_tbr = data['X_test_process'][query_idx]


    visualize_all(
        context, context_tbr, context_dbr, labels, 
        query, query_tbr, query_dbr, query_label,
        k=1024
    )


def visualize_all(context, context_tbr, context_dbr, labels, query, query_tbr, query_dbr, query_label, k=32):
    # FIXED: Handle label encoding properly
    from sklearn.preprocessing import LabelEncoder

    # Combine labels for encoding
    all_labels_str = np.concatenate((query_label, labels))
    
    # Encode labels
    # le = LabelEncoder()
    # all_labels_encoded = le.fit_transform(all_labels_str)
    
    # print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    # print(f"Unique labels in context: {np.unique(labels)}")

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
    process_retriever = ProcessRetriever(None, X_processes=context_tbr, pathway_column='pathway_with_prev_2_encoded')
    disease_retriever = TrajectoriesRetriever(None, X_traj=context_dbr)

    # Get neighbor indices
    neighbor_idxs_knn = retriever.retrieve(query, k=k)
    neighbor_idxs_process = process_retriever.retrieve(queries=None, queries_processes=query_tbr, k=k)
    neighbor_idxs_disease = disease_retriever.retrieve(queries=None, queries_traj=query_dbr, k=k)

    output_dir = '/home/mpham/workspace/source/ehrpfn/ehrpfn/xai/retrieval'
    os.makedirs(output_dir, exist_ok=True)
    np.save(osp.join(output_dir, 'reduced.npy'), reduced)
    np.save(osp.join(output_dir, 'neighbor_idxs_knn.npy'), neighbor_idxs_knn)
    np.save(osp.join(output_dir, 'neighbor_idxs_process.npy'), neighbor_idxs_process)
    np.save(osp.join(output_dir, 'neighbor_idxs_disease.npy'), neighbor_idxs_disease)
    np.save(osp.join(output_dir, 'all_labels_encoded.npy'), all_labels_str)


if __name__ == "__main__":
    # For single fold feature extraction
    visualize_retrieval()
