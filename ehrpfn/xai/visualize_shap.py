from typing import *
import numpy as np
import plotly.graph_objects as go
import pickle
import os.path as osp
import shap
from ehrpfn.datasets.data_mimic import collect_parquet, make_targets
import plotly.graph_objects as go
import numpy as np
import os

def extract_top_features_across_folds(
        task="SEPSIS", 
        root="/home/mpham/workspace/source/tabpfn-ehr", 
        n_folds=5, top_k=500
    ):
    """
    Extract top 500 features using different feature selection methods across all folds.
    
    Returns:
        dict: Dictionary with feature selection methods as keys and feature indices as values
    """
    
    # Storage for aggregated feature importance across folds
    xgb_importance_agg = None
    rf_importance_agg = None
    shap_importance_agg = None
    
    # Storage for individual fold rankings
    fold_rankings = {'xgboost': {}, 'random_forest': {}, 'shap': {}}
    
    feature_names = None
    
    for fold in range(1, n_folds + 1):
        print(f"Processing fold {fold}...")
        
        # Load XGBoost model
        xgb_model_path = f"{root}/data/tabpfn/results/{task}/xgboost_mimiciv_{task}/fold_{fold}/model.pkl"
        try:
            xgb_model = pickle.load(open(xgb_model_path, 'rb'))
        except FileNotFoundError:
            print(f"XGBoost model not found for fold {fold}, skipping...")
            continue
            
        # Load Random Forest model
        rf_model_path = f"{root}/data/tabpfn/results/{task}/rf_mimiciv_{task}/fold_{fold}/model.pkl"
        try:
            rf_model = pickle.load(open(rf_model_path, 'rb'))
        except FileNotFoundError:
            print(f"Random Forest model not found for fold {fold}, skipping...")
            continue
            
        # Load data for this fold
        data_folder = f"{root}/data/ehrfm/mimiciv-3.1/tokenized_datasets/mimiciv-icu-gbms-events"
        data_output_folder = f"{root}/data/tabpfn/data"
        data_input_folder = f"{data_folder}/train/09_FinalizeData"
        icu_id_path = f"{root}/data/ehrfm/mimiciv-3.1/icu_subject_ids.txt"
        train_label_path = f"{root}/data/tabpfn/labels/{task}/train_{task}_prediction_labels_fold_{fold}.csv"
        val_label_path = f"{root}/data/tabpfn/labels/{task}/val_{task}_prediction_labels_fold_{fold}.csv"
        
        with open(icu_id_path, 'r') as f:
            icu_patient_ids = list(map(lambda x: int(x.strip()), f.readlines()))
        
        # Collect training data
        X_train = collect_parquet(
            input_path=data_input_folder,
            output_path=data_output_folder,
            split="train",
            patient_ids_filter=icu_patient_ids,
            load_encoder_mapping=osp.join(data_output_folder, "3_encoding/train/encoder_mapping.pkl")
        )
        
        train_data, y_train, feature_names = make_targets(
            X_train,
            target_column=task,
            label_path=train_label_path
        )

        X_val = collect_parquet(
            input_path=data_input_folder,
            output_path=data_output_folder,
            split="val",
            patient_ids_filter=icu_patient_ids,
            load_encoder_mapping=osp.join(data_output_folder, "3_encoding/train/encoder_mapping.pkl")
        )
        
        val_data, y_val, feature_names = make_targets(
            X_val,
            target_column=task,
            label_path=val_label_path
        )

        all_data = np.concatenate((train_data[0], val_data[0]), axis=0)
        
        # 1. XGBoost feature importance
        xgb_importance = xgb_model.feature_importances_
        xgb_rankings = np.argsort(xgb_importance)[::-1]  # Rank by importance (0 = most important)
        fold_rankings['xgboost'][f'fold_{fold}'] = {
            'rankings': {feature_names[i]: rank for rank, i in enumerate(xgb_rankings)},
            'scores': {feature_names[i]: xgb_importance[i] for i in range(len(feature_names))}
        }
        
        if xgb_importance_agg is None:
            xgb_importance_agg = xgb_importance.copy()
        else:
            xgb_importance_agg += xgb_importance
            
        # 2. Random Forest feature importance
        rf_importance = rf_model.feature_importances_
        rf_rankings = np.argsort(rf_importance)[::-1]
        fold_rankings['random_forest'][f'fold_{fold}'] = {
            'rankings': {feature_names[i]: rank for rank, i in enumerate(rf_rankings)},
            'scores': {feature_names[i]: rf_importance[i] for i in range(len(feature_names))}
        }
        
        if rf_importance_agg is None:
            rf_importance_agg = rf_importance.copy()
        else:
            rf_importance_agg += rf_importance
            
        # 3. SHAP feature importance
        try:
            # Use a subset of data for SHAP to speed up computation
            sample_size = min(1000, len(all_data))
            sample_indices = np.random.choice(len(all_data), sample_size, replace=False)
            
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(all_data[sample_indices], check_additivity=False)
            shap_importance = np.abs(shap_values.values).mean(0)
            shap_rankings = np.argsort(shap_importance)[::-1]
            
            fold_rankings['shap'][f'fold_{fold}'] = {
                'rankings': {feature_names[i]: rank for rank, i in enumerate(shap_rankings)},
                'scores': {feature_names[i]: shap_importance[i] for i in range(len(feature_names))}
            }
            
            if shap_importance_agg is None:
                shap_importance_agg = shap_importance.copy()
            else:
                shap_importance_agg += shap_importance
        except Exception as e:
            print(f"SHAP calculation failed for fold {fold}: {e}")
    
    # Average importance scores across folds
    n_processed_folds = n_folds
    xgb_importance_agg /= n_processed_folds
    rf_importance_agg /= n_processed_folds
    if shap_importance_agg is not None:
        shap_importance_agg /= n_processed_folds
    
    # Get top features for each method
    top_features = {}
    
    # XGBoost top features
    xgb_top_indices = np.argsort(xgb_importance_agg)[::-1][:top_k]
    top_features['xgboost'] = {
        'indices': xgb_top_indices,
        'names': [feature_names[i] for i in xgb_top_indices],
        'scores': xgb_importance_agg[xgb_top_indices]
    }
    
    # Random Forest top features
    rf_top_indices = np.argsort(rf_importance_agg)[::-1][:top_k]
    top_features['random_forest'] = {
        'indices': rf_top_indices,
        'names': [feature_names[i] for i in rf_top_indices],
        'scores': rf_importance_agg[rf_top_indices]
    }
    
    # SHAP top features
    if shap_importance_agg is not None:
        shap_top_indices = np.argsort(shap_importance_agg)[::-1][:top_k]
        top_features['shap'] = {
            'indices': shap_top_indices,
            'names': [feature_names[i] for i in shap_top_indices],
            'scores': shap_importance_agg[shap_top_indices]
        }
    
    # Save results
    output_dir = f"{root}/data/tabpfn/feature_selection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated top features
    with open(f"{output_dir}/top_{top_k}_features_{task}.pkl", 'wb') as f:
        pickle.dump(top_features, f)
    
    # Save individual fold rankings
    with open(f"{output_dir}/fold_rankings_{task}.pkl", 'wb') as f:
        pickle.dump(fold_rankings, f)
        
    # Save as CSV for easy inspection
    import polars as pl
    
    # Save aggregated results
    for method, data in top_features.items():
        df = pl.DataFrame({
            'rank': range(1, len(data['indices']) + 1),
            'feature_index': data['indices'],
            'feature_name': data['names'],
            'importance_score': data['scores']
        })
        df.write_csv(f"{output_dir}/top_{top_k}_features_{method}_{task}.csv")
    
    # Save individual fold rankings as CSV
    for method, folds_data in fold_rankings.items():
        for fold, fold_data in folds_data.items():
            rankings_df = pl.DataFrame({
                'feature_name': list(fold_data['rankings'].keys()),
                'ranking': list(fold_data['rankings'].values()),
                'importance_score': [fold_data['scores'][feat] for feat in fold_data['rankings'].keys()]
            }).sort('ranking')
            
            rankings_df.write_csv(f"{output_dir}/{method}_{task}_{fold}_rankings.csv")
    
    print(f"Top {top_k} features and fold rankings saved to {output_dir}")
    return top_features, fold_rankings

def visualize_ranking_distributions(task="SEPSIS", root="/home/mpham/workspace/source/tabpfn-ehr", 
                                   top_k=20, remove_icd=True):
    """
    Create boxplot visualization of feature ranking distributions across folds and methods.
    """

    
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
    
    for method, folds_data in fold_rankings.items():
        for fold, fold_data in folds_data.items():
            for feature, rank in fold_data['rankings'].items():
                if feature not in all_rankings:
                    all_rankings[feature] = []
                all_rankings[feature].append(rank)
    
    # Filter features if needed
    features_with_data = list(all_rankings.keys())
    
    if remove_icd:
        features_with_data = [f for f in features_with_data if not any(icd in f.upper() for icd in ['ICD', 'DRG', 'MDC', 'TEXT'])]
        print(f"Filtered out ICD features, remaining: {len(features_with_data)}")
    
    # Calculate median rankings for sorting
    feature_medians = {}
    for feature in features_with_data:
        if all_rankings[feature]:
            feature_medians[feature] = np.median(all_rankings[feature])
    
    # Sort features by median ranking and take top_k (REVERSED SORTING - worst to best)
    feature_order = sorted(feature_medians.keys(), key=lambda x: feature_medians[x], reverse=False)[:top_k]
    
    # Color scheme for feature types
    colors = {
        'Current Episode': '#1f77b4',
        'Previous Episodes': '#2ca02c', 
        'Demographics': '#ff7f0e',
        'Network Features': '#d62728'
    }
    
    def get_feature_color(feature_name):
        """Simple feature type detection for coloring"""
        if 'WARD_' in feature_name or 'DEGREE' in feature_name or 'CENTRALITY' in feature_name:
            return colors['Network Features']
        elif 'SOFAR' in feature_name or 'PREVIOUS' in feature_name:
            return colors['Previous Episodes']
        elif feature_name in ['GENDER', 'AREA OF RESIDENCE', 'INSIDE DUBLIN', 'INSIDE IRELAND']:
            return colors['Demographics']
        else:
            return colors['Current Episode']
    
    # Create the boxplot
    fig = go.Figure()
    
    for i, feature in enumerate(feature_order):
        rankings_data = all_rankings[feature]
        color = get_feature_color(feature)
        
        # Simplify feature name for display
        display_name = feature.replace('_', ' ').title()
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        
        fig.add_trace(go.Box(
            x=rankings_data,
            y=[display_name] * len(rankings_data),
            boxpoints='outliers',
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
        # Limit x-axis to 500 and reverse the y-axis order
        xaxis=dict(range=[0, 300]),
        yaxis=dict(
            categoryorder='array',
            categoryarray=[feature.replace('_', ' ').title() for feature in feature_order]
        )
    )
    
    # Save the figure
    output_dir = f"{root}/data/tabpfn/feature_selection"
    pdf_path = f"{output_dir}/ranking_distributions_{task}_top{top_k}.pdf"
    fig.write_image(pdf_path, format='pdf')
   


if __name__ == '__main__':
    
    # Usage
    top_features, fold_rankings = extract_top_features_across_folds(task="SEPSIS", n_folds=5, top_k=500)

    # Visualize ranking distributions
    visualize_ranking_distributions(task="SEPSIS", top_k=20, remove_icd=True)