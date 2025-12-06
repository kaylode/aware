import os
import os.path as osp
import hydra
import numpy as np
import shap
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ehrpfn.utils import LoggerObserver
from dill import pickle
from ehrpfn.data import get_data
from ehrpfn.utils import load_model
from omegaconf import DictConfig

LOGGER = LoggerObserver.getLogger("main")


@hydra.main(version_base=None, config_path="../../configs", config_name="mimiciv")
def extract_top_features(args: DictConfig):
    """
    Extract top features using different feature selection methods for a single model.
    
    Args:
        args: Hydra configuration object containing all parameters
        
    Returns:
        tuple: (rankings_dict, scores_dict)
            - rankings_dict: Feature rankings
            - scores_dict: Feature importance scores
    """
    result_folder = osp.join(args.inference.output_folder, "selection", args.data.data_name, args.data.task)
    os.makedirs(result_folder, exist_ok=True)
    
    model_path = osp.join(args.explainable_model_path, 'model.pkl')

    if args.inference.model == 'xgboost':
        from xgboost import XGBClassifier, XGBRegressor
        if args.data.task in ['NEXT_ADMISSION_LOS', 'NEXT_LOS']:
            model = XGBRegressor()
        else:
            model = XGBClassifier()
        model = load_model(model_path, model)
    else:
        model = load_model(model_path)

    with open(osp.join(args.explainable_model_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    feature_names = metadata.get('feature_names', [])

    # Setup directories
    if args.get("use_shap", False):

        data = get_data(args)
        # Extract data components
        X_train = data['X_train']

        # Use a subset of data for SHAP to speed up computation
        # sample_size = min(1000, len(X_train))
        sample_indices = np.random.choice(len(X_train), len(X_train), replace=False)
        explainer = shap.TreeExplainer(model, feature_names=feature_names)
        shap_values = explainer(X_train[sample_indices], check_additivity=False)

        plt.clf()
        fig = shap.plots.beeswarm(shap_values, show=False, max_display=20)
        fig = plt.gcf()
        plt.savefig(osp.join(result_folder, 'shap_beeswarm.pdf'), bbox_inches='tight')
        importance = np.abs(shap_values.values).mean(0)


    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise ValueError(f"Method not supported")

    # Create rankings and scores
    rankings_idx = np.argsort(importance)[::-1]
    rankings = {feature_names[i]: rank for rank, i in enumerate(rankings_idx)}
    scores = {feature_names[i]: importance[i] for i in range(len(feature_names))}

    if args.get("use_shap", False):
        method = 'shap'
    else:
        method = args.inference.model

    # Save results
    with open(osp.join(result_folder, f"{method}_feature_importance.json"), 'w') as f:
        json.dump(rankings, f, indent=4)

    LOGGER.text(f"Feature importance saved to {result_folder}/{method}_feature_importance.json", level=LoggerObserver.INFO)

    return rankings, scores


def visualize_ranking_distributions(
    task="SEPSIS", 
    root="", 
    top_k=20, 
    remove_icd=True
):
    """Create boxplot visualization of feature ranking distributions across folds and methods."""
    # Load fold rankings
    rankings_path = f"{root}/data/tabpfn/feature_selection/fold_rankings_{task}.pkl"
    
    if not osp.exists(rankings_path):
        print(f"Rankings file not found: {rankings_path}")
        print("Run extract_top_features_across_folds() first")
        return
    
    with open(rankings_path, 'rb') as f:
        fold_rankings = pickle.load(f)
    
    # Collect all rankings by feature across methods and folds
    all_rankings = {}
    
    for fold, fold_data in fold_rankings.items():
        for feature, rank in fold_data['rankings'].items():
            if feature not in all_rankings:
                all_rankings[feature] = []
            all_rankings[feature].append(rank)
    
    # Filter features if needed
    features_with_data = list(all_rankings.keys())
    
    if remove_icd:
        icd_keywords = ['ICD', 'DRG', 'MDC', 'TEXT']
        features_with_data = [
            f for f in features_with_data 
            if not any(icd in f.upper() for icd in icd_keywords)
        ]
        print(f"Filtered out ICD features, remaining: {len(features_with_data)}")
    
    # Calculate median rankings for sorting
    feature_medians = {}
    for feature in features_with_data:
        if all_rankings[feature]:
            feature_medians[feature] = np.median(all_rankings[feature])
    
    # Sort features by median ranking and take top_k
    feature_order = sorted(feature_medians.keys(), key=lambda x: feature_medians[x])[:top_k]
    
    # Create the boxplot
    fig = _create_ranking_boxplot(all_rankings, feature_order, task, top_k)
    
    # Save the figure
    output_dir = f"{root}/data/tabpfn/feature_selection"
    _save_ranking_plot(fig, output_dir, task, top_k)


def _get_feature_color(feature_name):
    """Simple feature type detection for coloring."""
    colors = {
        'Current Episode': '#1f77b4',
        'Previous Episodes': '#2ca02c', 
        'Demographics': '#ff7f0e',
        'Network Features': '#d62728'
    }
    
    if 'WARD_' in feature_name or 'DEGREE' in feature_name or 'CENTRALITY' in feature_name:
        return colors['Network Features']
    elif 'SOFAR' in feature_name or 'PREVIOUS' in feature_name:
        return colors['Previous Episodes']
    elif feature_name in ['GENDER', 'AREA OF RESIDENCE', 'INSIDE DUBLIN', 'INSIDE IRELAND']:
        return colors['Demographics']
    else:
        return colors['Current Episode']


def _create_ranking_boxplot(all_rankings, feature_order, task, top_k):
    """Create the boxplot figure for ranking distributions."""
    fig = go.Figure()
    
    for i, feature in enumerate(feature_order):
        rankings_data = all_rankings[feature]
        color = _get_feature_color(feature)
        
        # Simplify feature name for display
        display_name = feature.replace('_', ' ').title()
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        
        fig.add_trace(go.Box(
            x=rankings_data,
            y=[display_name] * len(rankings_data),
            boxpoints='all',
            jitter=0.3,
            pointpos=0,
            marker_color=color,
            line_color=color,
            orientation='h',
            name=display_name,
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"Feature Ranking Distributions - {task} (Top {top_k})",
        xaxis_title="Feature Ranking (0 = Most Important)",
        yaxis_title="Features",
        template="plotly_white",
        height=max(400, len(feature_order) * 40),
        width=1000,
        margin=dict(l=250, r=50, t=80, b=50),
        yaxis=dict(
            categoryorder='array',
            categoryarray=[feature.replace('_', ' ').title() for feature in feature_order]
        )
    )
    
    return fig


def _save_ranking_plot(fig, output_dir, task, top_k):
    """Save the ranking plot as HTML and PDF."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = f"{output_dir}/ranking_distributions_{task}_top{top_k}.html"
    fig.write_html(fig_path)
    
    # Also save as PDF
    try:
        pdf_path = f"{output_dir}/ranking_distributions_{task}_top{top_k}.pdf"
        fig.write_image(pdf_path, format='pdf')
        print(f"Ranking distribution plots saved to:")
        print(f"  HTML: {fig_path}")
        print(f"  PDF: {pdf_path}")
    except Exception as e:
        print(f"HTML plot saved to: {fig_path}")
        print(f"Could not save PDF (install kaleido for PDF export): {e}")


def print_feature_summary(top_features):
    """Print summary of top features by each method."""
    print("\nTop 10 features by each method (aggregated across folds):")
    for method, data in top_features.items():
        print(f"\n{method.upper()}:")
        for i, (name, score) in enumerate(zip(data['names'][:10], data['scores'][:10])):
            print(f"  {i+1:2d}. {name}: {score:.4f}")


if __name__ == "__main__":
    # For single fold feature extraction
    extract_top_features()
    
    # For multi-fold feature extraction