from typing import Dict, List
import os
import os.path as osp
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np

TOP_K = 13
REMOVE_ICD = True  # Set to True to remove ICD features from the analysis

# Demographics '#bcbd22'
# Past episodes '#2ca02c',
# Contact Network '#e377c2'

INTERESTED_FEATURES = [
     'WARD_BETWEENNESS_CENTRALITY', 
     'WARD_INFECTED_CLOSENESS_CENTRALITY', 
     'WARD_CLUSTERING_COEFFICIENT', 'WARD_INFECTED_CONTACT_DAYS', 
     'WARD_INFECTED_DEGREE', 'WARD_INFECTED_CONTACT_DAYS_CENTRALITY', 
     'NUMBER_OF_CPE_SCREENS',
     'WARD_K_CORE_NUMBER', 'WARD_DEGREE', 'WARD_PAGERANK',
     'WARD_DEGREE_CENTRALITY',
     'WARD_CLOSENESS_CENTRALITY',
     'EPISODE_ISOLATED'
]
# INTERESTED_FEATURES = []

def get_ranking(attributions: List[Dict]):
    """
    Get the ranking of features based on their mean absolute attribution values.
    
    Args:
        attributions (Dict): Dictionary containing feature attributions.
        
    Returns:
        Dict: Dictionary mapping feature names to their rankings (0 = highest attribution).
    """
    # Group features by their base name (before //)
    mean_attributions = {}

    for pin_no, attribution_dict in attributions.items():
        for episode_id, attribution in attribution_dict.items():
            for feature, values in attribution.items():
                feature_group = feature.split('//')[0]
                if feature_group not in mean_attributions:
                    mean_attributions[feature_group] = []
                mean_attributions[feature_group].extend(values)
    
    # Calculate mean absolute attribution values for each feature group
    feature_means = {feature: np.abs(values).mean() for feature, values in mean_attributions.items()}
    
    # Sort features by their mean absolute attribution values in descending order
    sorted_features = sorted(feature_means, key=feature_means.get, reverse=True)
    
    # Return rankings (0 = highest attribution)
    return {feature: rank for rank, feature in enumerate(sorted_features)}


def simplify_feature_names(feature_names):
    """
    Simplify feature names for better visualization based on feature constants.
    """
    feature_mapping = {
        # Patient Demographics
        'GENDER': 'Gender',
        'AREA OF RESIDENCE': 'Area of Residence',
        'INSIDE DUBLIN': 'Inside Dublin',
        'INSIDE IRELAND': 'Inside Ireland',
        
        # Current Episode - Discrete
        'ADMISSION WARD': 'Admission Ward',
        'DISCHARGE WARD': 'Discharge Ward',
        'DISCHARGE DESCRIPTION': 'Discharge Description',
        'DISCHARGE SPECIALTY': 'Discharge Specialty',
        
        # Current Episode - Continuous
        'NUMBER_OF_CPE_SCREENS': 'CPE Screens',
        'EPISODE_ISOLATED': 'Episode Isolated',
        'LOS': 'Length of Stay',
        'NUM_DIAGNOSIS': 'No. Diagnoses',
        'NUM_PROCEDURES': 'No. Procedures',
        'NUM_WARDS': 'No. Wards',
        'EPISODE_NUMBER_EMERGENCY_DEPARTMENT_VISITS': 'ED Visits',
        'EPISODE_NUMBER_ACUTE_DEPARTMENT_VISITS': 'Acute Visits',
        
        # Current Episode - Codes
        'ICD10CMs_5': 'ICD10CM (Full)',
        'ICD10CMs_3': 'ICD10CM (Grouped)',
        'ICD10PCSs_9': 'ICD10PCS (Full)',
        'ICD10PCSs_5': 'ICD10PCS (Grouped)',
        'DRG_CODE': 'DRG Code',
        'MDC_CODE': 'MDC Code',
        
        # Current Episode - Text/NLP
        'COMBINED_TEXT': 'ICD Description',
        'PATHWAYS': 'Pathways',
        
        # Previous Episodes - Discrete
        'DISCHARGE DESCRIPTION_SOFAR': 'Previous Discharge Desc.',
        'DISCHARGE SPECIALTY_SOFAR': 'Previous Discharge Spec.',
        
        # Previous Episodes - Continuous
        'NUMBER_OF_CPE_SCREENS_SOFAR': 'No. Previous CPE Screens',
        'READMISSION_SOFAR': 'No. Previous Readmissions',
        'LOS_SOFAR': 'Previous LOS Total',
        'NUM_DIAGNOSIS_SOFAR': 'No. Previous Diagnoses',
        'NUM_PROCEDURES_SOFAR': 'No. Previous Procedures',
        'NUM_WARDS_SOFAR': 'No. Previous Wards',
        'EPISODE_NUMBER_EMERGENCY_DEPARTMENT_VISITS_SOFAR': 'No. Previous ED Visits',
        'EPISODE_NUMBER_ACUTE_DEPARTMENT_VISITS_SOFAR': 'No. Previous Acute Visits',
        
        # Previous Episodes - Codes
        'ICD10CMs_5_SOFAR': 'Previous ICD10CM (Full)',
        'ICD10CMs_3_SOFAR': 'Previous ICD10CM (Grouped)',
        'ICD10PCSs_9_SOFAR': 'Previous ICD10PCS (Full)',
        'ICD10PCSs_5_SOFAR': 'Previous ICD10PCS (Grouped)',
        'DRG_CODE_SOFAR': 'Previous DRG Codes',
        'MDC_CODE_SOFAR': 'Previous MDC Codes',
        
        # Previous Episodes - Text/NLP
        'COMBINED_TEXT_SOFAR': 'Previous ICD Description',
        'PATHWAYS_SOFAR': 'Previous Pathways',
        
        # Network Features
        'WARD_DEGREE_CENTRALITY': 'Ward Degree Centrality',
        'WARD_INFECTED_DEGREE': 'Ward Infected Degree',
        'WARD_INFECTED_CONTACT_DAYS': 'Ward Infected Contact Days',
        'WARD_INFECTED_CONTACT_DAYS_CENTRALITY': 'Ward Infected Contact Centrality',
        'WARD_CLUSTERING_COEFFICIENT': 'Ward Clustering Coefficient',
        'WARD_CLOSENESS_CENTRALITY': 'Ward Closeness Centrality',
        'WARD_INFECTED_CLOSENESS_CENTRALITY': 'Ward Infected Closeness',
        'WARD_BETWEENNESS_CENTRALITY': 'Ward Betweenness Centrality',
        'WARD_PAGERANK': 'Ward PageRank',
        'WARD_K_CORE_NUMBER': 'Ward K-Core Number',
        'WARD_DEGREE': 'Ward Degree',
        
        # Additional
        'TIME_SINCE_LAST_ADMISSION': 'Days Since Last Admission',
        'EPISODE_CPE_SCREEN': 'CPE Screened',
    }
    
    simplified_names = []
    for feature in feature_names:
        # Use mapping if available, otherwise use original name with some cleanup
        if feature in feature_mapping:
            simplified_names.append(feature_mapping[feature])
        else:
            # Basic cleanup: replace underscores with spaces and title case
            simplified = feature.replace('_', ' ').title()
            simplified_names.append(simplified)
    
    return simplified_names

def get_feature_type_and_color(feature_name):
    """
    Determine the feature type and corresponding color based on feature name.
    
    Returns:
        tuple: (feature_type, color)
    """
    # Define feature type mappings
    demographics_features = [
        'GENDER', 'AREA OF RESIDENCE', 'INSIDE DUBLIN', 'INSIDE IRELAND',
        'Gender', 'Area of Residence', 'Inside Dublin', 'Inside Ireland'
    ]
    
    network_features = [
        'WARD_DEGREE_CENTRALITY', 'WARD_INFECTED_DEGREE', 'WARD_INFECTED_CONTACT_DAYS',
        'WARD_INFECTED_CONTACT_DAYS_CENTRALITY', 'WARD_CLUSTERING_COEFFICIENT',
        'WARD_CLOSENESS_CENTRALITY', 'WARD_INFECTED_CLOSENESS_CENTRALITY',
        'WARD_BETWEENNESS_CENTRALITY', 'WARD_PAGERANK', 'WARD_K_CORE_NUMBER', 'WARD_DEGREE',
        'Ward Degree Centrality', 'Ward Infected Degree', 'Ward Infected Contact Days',
        'Ward Infected Contact Centrality', 'Ward Clustering Coefficient',
        'Ward Closeness Centrality', 'Ward Infected Closeness',
        'Ward Betweenness Centrality', 'Ward PageRank', 'Ward K-Core Number', 'Ward Degree'
    ]
    
    previous_episodes_features = [
        'DISCHARGE DESCRIPTION_SOFAR', 'DISCHARGE SPECIALTY_SOFAR', 'NUMBER_OF_CPE_SCREENS_SOFAR',
        'READMISSION_SOFAR', 'LOS_SOFAR', 'NUM_DIAGNOSIS_SOFAR', 'NUM_PROCEDURES_SOFAR',
        'NUM_WARDS_SOFAR', 'EPISODE_NUMBER_EMERGENCY_DEPARTMENT_VISITS_SOFAR',
        'EPISODE_NUMBER_ACUTE_DEPARTMENT_VISITS_SOFAR', 'ICD10CMs_5_SOFAR', 'ICD10CMs_3_SOFAR',
        'ICD10PCSs_9_SOFAR', 'ICD10PCSs_5_SOFAR', 'DRG_CODE_SOFAR', 'MDC_CODE_SOFAR',
        'COMBINED_TEXT_SOFAR', 'PATHWAYS_SOFAR', 'TIME_SINCE_LAST_ADMISSION',
        'Previous Discharge Desc.', 'Previous Discharge Spec.', 'No. Previous CPE Screens',
        'No. Previous Readmissions', 'Previous LOS Total', 'No. Previous Diagnoses',
        'No. Previous Procedures', 'No. Previous Wards', 'No. Previous ED Visits',
        'No. Previous Acute Visits', 'Previous ICD10CM (Full)', 'Previous ICD10CM (Grouped)',
        'Previous ICD10PCS (Full)', 'Previous ICD10PCS (Grouped)', 'Previous DRG Codes',
        'Previous MDC Codes', 'Previous ICD Description', 'Previous Pathways',
        'Days Since Last Admission'
    ]
    
    # Color scheme
    colors = {
        'Demographics': '#ff7f0e',
        'Network Features': '#d62728', 
        'Previous Episodes': '#2ca02c',
        'Current Episode': '#1f77b4'  # Default color for current episode features
    }

    # '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    
    # Check feature type
    if any(demo in feature_name for demo in demographics_features):
        return 'Demographics', colors['Demographics']
    elif any(network in feature_name for network in network_features):
        return 'Network Features', colors['Network Features']
    elif any(prev in feature_name for prev in previous_episodes_features):
        return 'Previous Episodes', colors['Previous Episodes']
    else:
        return 'Current Episode', colors['Current Episode']

def run(task_name:str, top_k:int=None):
    all_rankings = {}
    
    if task_name == 'all':
        baseline_modes = ["mean", "mode", "zero"]
        for task in ["WILL_READMIT_WITHIN_30_DAYS", "NEXT_LOS", "DECEASED_NEXT_EPISODE", "EPISODE_CPE_INFECTION_FLAGGED"]:
            attribution_folder = f"/home/mpham/workspace/source/carbapen/outputs/sjh-cpe/xai/attributions/{task}/feat_attributions"
            for baseline_mode in baseline_modes:
                folder = osp.join(attribution_folder, baseline_mode)
                if not osp.exists(folder):
                    print(f"Warning: Folder {folder} does not exist, skipping...")
                    continue
                    
                json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
                print(f"Processing {len(json_files)} files from {baseline_mode} baseline for task {task}...")
                
                for json_file in json_files:
                    # Load the attribution data
                    file_path = osp.join(folder, json_file)
                    with open(file_path, 'r') as f:
                        attributions = json.load(f)
                    rankings = get_ranking(attributions)

                    # Store the rankings in a dictionary with the feature as the key
                    for feature, rank in rankings.items():
                        if task == 'WILL_READMIT_WITHIN_30_DAYS' and feature == 'NEXT_ADMISSION_WARD': continue
                        if task == 'DECEASED_NEXT_EPISODE' and feature == 'DISCHARGE DESCRIPTION': continue
                        if feature in ['INSIDE IRELAND', 'INSIDE DUBLIN']: continue
                        if feature not in all_rankings:
                            all_rankings[feature] = []
                        all_rankings[feature].append(rank)
                            
    else:

        attribution_folder = f"/home/mpham/workspace/source/carbapen/outputs/sjh-cpe/xai/attributions/{task_name}/feat_attributions"
        baseline_modes = ["mean", "mode", "zero"]
        
        for baseline_mode in baseline_modes:
            folder = osp.join(attribution_folder, baseline_mode)
            if not osp.exists(folder):
                print(f"Warning: Folder {folder} does not exist, skipping...")
                continue
                
            json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
            print(f"Processing {len(json_files)} files from {baseline_mode} baseline...")
            
            for json_file in json_files:
                # Load the attribution data
                file_path = osp.join(folder, json_file)
                with open(file_path, 'r') as f:
                    attributions = json.load(f)
                rankings = get_ranking(attributions)

                # Store the rankings in a dictionary with the feature as the key
                for feature, rank in rankings.items():
                    if task_name == 'WILL_READMIT_WITHIN_30_DAYS' and feature == 'NEXT_ADMISSION_WARD': continue
                    if task_name == 'DECEASED_NEXT_EPISODE' and feature == 'DISCHARGE DESCRIPTION': continue
                    if feature in ['INSIDE IRELAND', 'INSIDE DUBLIN']: continue
                    if feature not in all_rankings:
                        all_rankings[feature] = []
                    all_rankings[feature].append(rank)
                    

        if not all_rankings:
            print("No rankings data found!")
            return

    # Instead of DataFrame, work with lists directly
    print("Processing rankings data...")
    
    # Get all features that have at least some data
    features_with_data = list(all_rankings.keys())
    print(f"Found {len(features_with_data)} features with ranking data")
    
    # Calculate median rankings for sorting (handle different lengths)
    feature_medians = {}
    for feature, rankings in all_rankings.items():
        if rankings:  # Only if we have data
            feature_medians[feature] = np.median(rankings)
    
    # Sort features by their median ranking
    feature_order = sorted(feature_medians.keys(), key=lambda x: feature_medians[x], reverse=True)
    
    # Remove ICD features if specified
    if REMOVE_ICD:
        feature_order = [f for f in feature_order if not any(icd in f for icd in ['ICD', 'DRG', 'MDC', 'TEXT'])]
        print(f"Filtered out ICD features, remaining: {len(feature_order)}")

    # Apply top-k filtering if specified
    if top_k is not None and top_k > 0:
        feature_order = feature_order[-top_k:]
        print(f"Filtered to top {top_k} features based on median ranking")
    else:
        if INTERESTED_FEATURES:
            # Filter to only include interested features if specified
            feature_order = [f for f in feature_order if f in INTERESTED_FEATURES]
            print(f"Filtered to interested features, remaining: {len(feature_order)}")


    # Simplify feature names for visualization
    simplified_feature_names = simplify_feature_names(feature_order)
    feature_name_mapping = dict(zip(feature_order, simplified_feature_names))

    # Create Plotly Box Plots (Horizontal)
    fig = go.Figure()

    # Keep track of which feature types we've seen for legend
    legend_added = set()

    for i, feature in enumerate(feature_order):
        simplified_name = feature_name_mapping[feature]
        rankings_data = all_rankings[feature]
        
        # Get feature type and color
        feature_type, color = get_feature_type_and_color(feature)
        
        # Only show legend for the first occurrence of each feature type
        show_legend = feature_type not in legend_added
        if show_legend:
            legend_added.add(feature_type)
        
        fig.add_trace(go.Box(
            x=rankings_data,
            y=[simplified_name] * len(rankings_data),
            name=feature_type,  # Use feature type as name for legend grouping
            boxpoints='all',
            jitter=0.3,
            pointpos=0,
            marker_color=color,
            line_color=color,
            orientation='h',
            legendgroup=feature_type,
            showlegend=show_legend  # Only show legend for first occurrence
        ))

    fig.update_layout(
        title=None,
        xaxis_title="Feature Ranking (0 = Most Important)",
        yaxis_title="Features",
        template="plotly_white",
        height=max(400, len(feature_order) * 50),
        width=900,
        legend=dict(
            orientation="h",     # Horizontal orientation
            x=0.4,              # Center horizontally
            y=1.1,             # Position below the plot
            xanchor="center",   # Center anchor point
            yanchor="top",      # Top anchor point
            title="Feature Types",
            itemsizing="constant"  # Make legend items same size
        ),
        yaxis=dict(
            categoryorder='array', 
            categoryarray=simplified_feature_names,
            tickmode='array',
            tickvals=list(range(len(simplified_feature_names))),
            ticktext=simplified_feature_names
        ),
        margin=dict(l=200, r=50, t=50, b=100),  # Increase bottom margin for horizontal legend
        boxgap=0.1,
        boxgroupgap=0.1,
    )
    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Reverse x-axis so lower rankings (more important) appear on the left
    fig.update_xaxes(autorange="reversed")

    # Save the figure
    output_dir = "/home/mpham/workspace/source/carbapen/outputs/sjh-cpe/xai/attributions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with top_k suffix if applicable
    filename_suffix = f"_top{top_k}" if top_k is not None else ""
    filename_suffix += f"_remove_icd" if REMOVE_ICD else ""
    filename_suffix += f"_cpe" if INTERESTED_FEATURES else ""
    pdf_path = osp.join(output_dir, task_name, 'feat_attributions', f"boxplot_rankings_{task_name}{filename_suffix}.pdf")
    
    fig.write_image(pdf_path, format='pdf')
    
    print(f"Plots saved to:")
    print(f"  PDF: {pdf_path}")
    
if __name__ == "__main__":

    for TASK_NAME in ["WILL_READMIT_WITHIN_30_DAYS", "NEXT_LOS", "DECEASED_NEXT_EPISODE", "EPISODE_CPE_INFECTION_FLAGGED"]:
        run(TASK_NAME, TOP_K)

    run("all", None)