import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from ehrpfn.datasets import get_mimic_dataset_v2
from ehrpfn.representation import SDAETransformer
from ehrpfn.utils.retrieval import FAISS
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
from ehrpfn.utils.logger import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')


def evaluate_embeddings(emb_train, emb_test, y_train, y_test):
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(emb_train, y_train)
    y_pred_knn = knn.predict(emb_test)
    auc_score = roc_auc_score(y_test, y_pred_knn)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    mlp.fit(emb_train, y_train)
    y_pred_mlp = mlp.predict(emb_test)
    mlp_auc = roc_auc_score(y_test, y_pred_mlp)

    return auc_score, mlp_auc

def visualize_embedding(all_train, all_test, y_train, y_test, vis_train, vis_test, k=32, method_name="SDAE", save_dir:str=None):
    retriever = FAISS(all_train)
    neighbor_idxs_knn = retriever.retrieve(all_test, k=k)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=250)
    combined_data = np.concatenate((vis_test, vis_train), axis=0)
    reduced = tsne.fit_transform(combined_data)

    query_shape = neighbor_idxs_knn.shape[0]
    query_items = reduced[:query_shape]
    context_items = reduced[query_shape:]

    # Plot neighbors for the first query
    fig_neighbors = plot_neighbors_plotly(
        context_items, y_train, neighbor_idxs_knn, query_items, y_test, query_idx=0, method_name=method_name
    )
    if save_dir:
        fig_neighbors.write_image(f"{save_dir}/neighbors_query_0.pdf",format='pdf')
        fig_neighbors.write_html(f"{save_dir}/neighbors_query_0.html")

def plot_context_label_distribution(context_labels):
    from collections import Counter

    label_counts = Counter(context_labels)
    infection_order = ['ICD BSI', 'ICD SEPSIS', 'ICD UTI', 'ICD VAP', 'ICD SSI', 'Readmission', 'Length of Stay', 'Mortality', 'None']
    counts = [label_counts.get(label, 0) for label in infection_order]
    labels = infection_order

    total = sum(counts)
    percentages = [count/total * 100 if total > 0 else 0 for count in counts]

    colors = px.colors.qualitative.Bold
    color_map = {
        'SEPSIS': colors[4],
        'BSI': colors[1],
        'VAP': colors[3],
        'SSI': colors[2],
        'UTI': colors[0],
        'ICD SEPSIS': colors[4],
        'ICD BSI': colors[1],
        'ICD VAP': colors[3],
        'ICD SSI': colors[2],
        'ICD UTI': colors[0],
        'Readmission': colors[5],
        'Length of Stay': colors[6],
        'Mortality': colors[7],
        'None': colors[-1],
    }

    colors = [color_map[label] for label in labels]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=counts,
            marker_color=colors,
            text=[f'{count}<br>({pct:.1f}%)' for count, pct in zip(counts, percentages)],
            textposition='auto',
            textfont=dict(size=14, color='white')
        )
    ])

    fig.update_layout(
        title='Context Labels Distribution',
        xaxis_title='Infection Type',
        yaxis_title='Count',
        template='plotly_white',
        font=dict(size=12),
        width=700,
        height=400,
        showlegend=False
    )

    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>' +
                      'Count: %{y}<br>' +
                      'Percentage: %{text}<br>' +
                      '<extra></extra>'
    )

    return fig

def plot_neighbors_plotly(context_items, context_labels, neighbor_idxs, query_items, query_labels, query_idx, method_name):
    viz_df = pd.DataFrame({
        'x': context_items[:, 0],
        'y': context_items[:, 1],
        'label': context_labels,
    })

    colors = px.colors.qualitative.Bold
    color_map = {
        'SEPSIS': colors[4],
        'BSI': colors[1],
        'VAP': colors[3],
        'SSI': colors[2],
        'UTI': colors[0],
        'ICD SEPSIS': colors[4],
        'ICD BSI': colors[1],
        'ICD VAP': colors[3],
        'ICD SSI': colors[2],
        'ICD UTI': colors[0],
        'Readmission': colors[5],
        'Length of Stay': colors[6],
        'Mortality': colors[7],
        'None': colors[-1],
    }

    # Main scatter plot for all context points
    fig = go.Figure()
    for label in color_map:
        mask = viz_df['label'] == label
        fig.add_trace(go.Scatter(
            x=viz_df.loc[mask, 'x'],
            y=viz_df.loc[mask, 'y'],
            mode='markers',
            marker=dict(color=color_map[label], size=9, opacity=0.7),
            name=label,
            showlegend=True
        ))

    # Highlight nearest neighbors
    fig.add_trace(go.Scatter(
        x=context_items[neighbor_idxs[query_idx], 0],
        y=context_items[neighbor_idxs[query_idx], 1],
        mode='markers',
        name=f'Top-{len(neighbor_idxs[query_idx])} Neighbors',
        marker=dict(
            size=10,
            color='rgba(255,255,255,0.1)',
            line=dict(color='black', width=2)
        ),
        customdata=neighbor_idxs[query_idx],
        showlegend=True,
    ))

    # Highlight query point
    query_item = query_items[query_idx]
    query_label = query_labels[query_idx]
    fig.add_trace(go.Scatter(
        x=[query_item[0]],
        y=[query_item[1]],
        mode='markers',
        name=f'Query ({query_label})',
        marker=dict(
            size=22,
            color='gold',
            symbol='star',
            line=dict(color='black', width=2)
        ),
        showlegend=True,
    ))

    fig.update_layout(
        title=f'{method_name} Retrieval Space',
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2',
        legend_title='Infection Type',
        font=dict(size=12),
        template='plotly_white',
        width=800,
        height=600,
        legend=dict(itemsizing='constant')
    )

    return fig

def main():
    # parser = argparse.ArgumentParser(description="Evaluate embeddings on a downstream task")
    # parser.add_argument("--root", type=str, required=True, help="Project root directory")
    # parser.add_argument("--task", type=str, required=True, help="Task name (e.g., ICD SEPSIS)")
    # args = parser.parse_args()

    ROOT="/home/mpham/workspace/source/ehrpfn"
    TASK="ICD SEPSIS"

    data_csv = f"{ROOT}/data/processed/mimiciv/{TASK}/all_data.csv"
    train_label_path = f"{ROOT}/data/processed/mimiciv/{TASK}/split/train_labels.csv"
    test_label_path = f"{ROOT}/data/processed/mimiciv/{TASK}/split/test_labels.csv"
    feature_selection_path = f"{ROOT}/outputs/tabpfn/selection/mimicpipe/{TASK}"

    data = get_mimic_dataset_v2(
        data_csv=data_csv,
        train_label_path=train_label_path,
        test_label_path=test_label_path,
        feature_selection_path=feature_selection_path,
        selection_method=None,
        process_file_path=None,
        trajectory_file_path=None,
    )

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    LOGGER.text("Train shape:", X_train.shape)
    LOGGER.text("Test shape:", X_test.shape)


    pipeline = SDAETransformer.load(
        "/home/mpham/workspace/source/ehrpfn/outputs/pretraining/ptsdae/ICD SEPSIS/sdae2.pt",
    )

    all_train = pipeline.transform(X_train)
    all_test = pipeline.transform(X_test)

    LOGGER.text("Embedded train shape:", all_train.shape)
    LOGGER.text("Embedded test shape:", all_test.shape)

    evaluate_embeddings(all_train, all_test, y_train, y_test)

    # Qualitative evaluation
    visualize_embedding(
        all_train, all_test, 
        y_train, y_test, 
        X_train, X_test, 
        k=32, method_name="SDAE", 
        save_dir=save_path
    )

if __name__ == "__main__":
    main()
  
