import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(42)
np.random.seed(42)

names_dict = {
    "dummy": "Dummy",
    "LogReg": "LR",
    "logistics": "LR",
    "LinearRegression": "LR",
    "NCM": 'NCM',
    "NaiveBayes": "NB",
    "knn": "KNN",
    "svm": "SVM",
    "xgboost": "XGB",
    "catboost": "CatB",
    "RandomForest": "RForest",
    "lightgbm": "LightG",
    "tabpfn": "TabPFN",
    "mlp": "MLP",
    "resnet": "ResNet",
    "node": "NODE",
    "switchtab": "SwitchT",
    "tabnet": "TabNet",
    "tabcaps": "TabCaps",
    "tangos": "TANGOS",
    "danets": "DANets",
    "ftt": "FT-T",
    "autoint": "AutoInt",
    "dcn2": "DCNv2",
    "snn": "SNN",
    "tabtransformer": "TabT",
    "ptarl": "PTaRL",
    "grownet": "GrowNet",
    "tabr": "TabR",
    "dnnr": "DNNR",
    "realmlp": "RealMLP",
    "mlp_plr": "MLP-PLR",
    "excelformer": "ExcelF",
    "modernNCA": "MNCA",
    "tabm": "TabM",
    "tabdpt_knn": "TabDPT",
    "tabicl": "TabICL",
    'tabpfn-retrieval_knn': 'kNNPFN',
    'trompt': 'Trompt',
    'beta': 'BETA',
    'corebehrt': 'CoreBEHRT'
}

method_types={
    "A": ["MNCA", "TabR"], # Retrieval-based
    "B": ["LR", "NCM", "NB", "KNN", "SVM", "DNNR"], # Classical ML
    "C": ["XGB", "CatB", "RForest", "LightG"], # GBMS
    "D": ["MLP", "SNN", "MLP-PLR", 'RealMLP', "ResNet"], # Deep Learning
    # "E": ["DCNv2", "DANets", "TabCaps"],
    # "F": ["NODE", "GrowNet"],
    "G": ["TabPFN",  "TabICL", 'TabDPT', 'kNNPFN', 'BETA'], # PFN-based
    "H": ["FT-T", 'TabNet', "AutoInt", "ExcelF", "TabT", 'Trompt', 'TabM'], # Transformer-based
    # "I": ["SwitchT", "TANGOS", "PTaRL"]
    "I": ['CoreBEHRT']
}

colors_type = {
    "A": "#D4E157",  # Fresh Lime
    "I": "#FFFFFF",  # White
    "B": "#FF7043",  # Coral Orange
    "C": "#66BB6A",  # Vibrant Green
    "D": "#EF5350",  # Rich Red
    "E": "#5C6BC0",  # Soft Indigo
    "F": "#26A69A",  # Emerald Teal
    "G": "#AB47BC",  # Vivid Purple
    "H": "#26C6DA",  # Bright Cyan
    "I": "#5C6BC0",  # Soft Indigo
}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['Arial']


def calculate_elo_scores_with_missing_values(df, k_factor=32, initial_elo=1500):
    """
    Calculate ELO scores for multiple methods based on their performance across datasets,
    skipping rows with missing values for any method pair.

    Parameters:
    - df: DataFrame where each row is a dataset and each column is a method's performance.
    - k_factor: K-factor for the ELO calculation.
    - initial_elo: Initial ELO score for each method.

    Returns:
    - elo_scores: A dictionary of final ELO scores for each method.
    """
    methods = df.columns
    elo_scores = {method: initial_elo for method in methods}
    
    for _, row in df.iterrows():
        # Skip rows with missing values
        if row.isnull().any():
            continue
        performances = row
        
        # Compare each pair of methods
        for method_a in methods:
            for method_b in methods:
                if method_a == method_b:
                    continue

                rating_a = elo_scores[method_a]
                rating_b = elo_scores[method_b]
                performance_a = performances[method_a]
                performance_b = performances[method_b]

                # Determine the match result
                if performance_a > performance_b:
                    result_a, result_b = 1, 0
                elif performance_a < performance_b:
                    result_a, result_b = 0, 1
                else:
                    result_a, result_b = 0.5, 0.5

                # Calculate expected scores
                expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

                # Update ELO scores
                elo_scores[method_a] += k_factor * (result_a - expected_a)
                elo_scores[method_b] += k_factor * (result_b - expected_b)

    return elo_scores


def calculate_average_elo_scores(df, k_factor=32, initial_elo=1500, num_shuffles=30):
    """
    Calculate average ELO scores for multiple methods across multiple randomized runs.

    Parameters:
    - df: DataFrame where each row is a dataset and each column is a method's performance.
    - k_factor: K-factor for the ELO calculation.
    - initial_elo: Initial ELO score for each method.
    - num_shuffles: Number of times to shuffle the datasets and calculate ELO.

    Returns:
    - avg_elo_scores: A dictionary of average ELO scores for each method across all shuffles.
    """
    methods = df.columns
    accumulated_elo_scores = {method: 0 for method in methods}

    for _ in range(num_shuffles):
        # Shuffle the datasets (rows)
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        # Calculate ELO scores for this shuffled order
        elo_scores = calculate_elo_scores_with_missing_values(shuffled_df, k_factor, initial_elo)
        
        # Accumulate ELO scores from this run
        for method in methods:
            accumulated_elo_scores[method] += elo_scores[method]

    # Calculate the average ELO scores
    avg_elo_scores = {method: score / num_shuffles for method, score in accumulated_elo_scores.items()}
    return avg_elo_scores


def plot_elo_scores(elo_scores, path='pics/elo_scores.pdf'):
    """
    Plot ELO scores as a horizontal bar chart, with mean and std lines.
    """
    sorted_elo_scores = dict(sorted(elo_scores.items(), key=lambda item: item[1], reverse=True))
    methods = list(sorted_elo_scores.keys())
    scores = np.array(list(sorted_elo_scores.values()))

    plt.figure(figsize=(10, 8))
    plt.xlim(min(scores) - 50, max(scores) + 500)
    colors = []
    for model in methods:
        for key, value in method_types.items():
            if model in value:
                colors.append(colors_type[key])
                break

    plt.barh(methods, scores, color=colors, edgecolor='black', alpha=0.5)
    plt.xlabel('ELO Score', fontsize=18)
    plt.ylabel('Method', fontsize=18)
    plt.gca().invert_yaxis()

    # Annotate each bar with its ELO score
    for index, value in enumerate(scores):
        plt.text(value + 5, index, f"{value:.2f}", va='center', fontsize=8)

    # Add mean and std lines
    mean = scores.mean()
    std = scores.std()
    plt.axvline(mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean ({mean:.2f})')
    plt.axvline(mean + std, color='grey', linestyle=':', linewidth=1, label=f'Mean+Std ({mean+std:.2f})')
    plt.axvline(mean - std, color='grey', linestyle=':', linewidth=1, label=f'Mean-Std ({mean-std:.2f})')
    plt.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(path)


def plot_elo_scores_with_std(global_elo_scores, path='elo_global.pdf'):
    """
    Plot global ELO scores as a horizontal bar chart with error bars (std for each method),
    and add a color legend for method types.
    """
    # Calculate mean and std for each method
    methods = list(global_elo_scores.keys())
    means = np.array([np.mean(global_elo_scores[m]) for m in methods])
    stds = np.array([np.std(global_elo_scores[m]) for m in methods])

    # Sort by mean ELO score
    sorted_idx = np.argsort(means)[::-1]
    methods = [methods[i] for i in sorted_idx]
    means = means[sorted_idx]
    stds = stds[sorted_idx]

    plt.figure(figsize=(10, 8))
    plt.xlim(0, max(means) + 500)
    colors = []
    type_labels = []
    for model in methods:
        for key, value in method_types.items():
            if model in value:
                colors.append(colors_type[key])
                type_labels.append(key)
                break

    bars = plt.barh(methods, means, xerr=stds, color=colors, edgecolor='black', alpha=0.5, capsize=5)
    plt.xlabel('ELO Score', fontsize=18)
    plt.ylabel('Method', fontsize=18)
    plt.gca().invert_yaxis()

    # Annotate each bar with its mean ELO score
    for index, value in enumerate(means):
        plt.text(value + stds[index] + 5, index, f"{value:.2f}±{stds[index]:.2f}", va='center', fontsize=8)

    # Build legend handles
    from matplotlib.patches import Patch
    legend_elements = []
    type_comments = {
        "A": "Retrieval-based",
        "B": "Classical ML",
        "C": "GBMS",
        "D": "Deep Learning",
        "G": "PFN-based",
        "H": "Transformer-based",
        "I": "Sequential / EHR-specific"
    }
    for key in method_types.keys():
        legend_elements.append(Patch(facecolor=colors_type[key], edgecolor='black', label=f"{type_comments[key]}"))

    plt.legend(
        handles=legend_elements,
        loc='lower right',
        bbox_to_anchor=(1.45, 0),  # Move legend further right
        ncol=1,
        fontsize=11,
        title="Method Type",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(path)


def get_metric_for_task(task_name, is_classification):
# "Name","MODEL.MODEL_NAME","task_name","dataset_name","Tags","Created","auc","auprc","rmse","mae","bl_acc","f1","precision","recall","is_classification","inference_time","Runtime","End Time"

    # Define your logic for metric selection based on task_name
    if is_classification:
        if task_name in ['Diabetes130US', 'COVID-19-Hospitals-Treatment-Plan']:
            return 'f1'
        else:
            return 'auc'
    else:
        return 'mae'


def prepare_elo_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    tasks = df['task_name'].unique()
    tasks = [task for task in tasks if task not in [
        'Dataset-Mental-Disorders', 'postoperative-patient-data',
        'ICD UTI', 'heart_failure_clinical_records', 'parkinsons-telemonitoring',
        'COVID-19-Hospitals-Treatment-Plan', 'ICU'
    ]]
    dfs = {}
    for task in tasks:
        df_task = df[df['task_name'] == task]
        is_classification = df_task['is_classification'].iloc[0]
        metric = get_metric_for_task(task, is_classification)

        df_task = df_task[['Name', 'task_name', metric]]
        df_task = df_task[df_task[metric].notnull()]
        df_pivot = df_task.pivot_table(index='task_name', columns='Name', values=metric, aggfunc='mean')
        df_pivot = df_pivot.rename(columns=names_dict)
        # Reverse ranking for regression (lower is better)
        if not is_classification and metric == 'mae':
            df_pivot = -df_pivot

        try:
            df_pivot = df_pivot.drop([
                'tabdpt_weighted_logistics',
                'tabdpt_weighted_feattype',
                'ehrpfn-mimiciv-synthetic',
                'tabpfn-retrieval_weighted_logistics',
                'tabpfn-retrieval_weighted_feattype',
                'knn-simple_weighted_logistics',
                'knn-simple_knn',
                'knn-simple_weighted_feattype',
                'knn-simple_dbr',
                'tabdpt_random',
                'ehrpfn500',
                'ehrpfn-finetune',
                'ehrpfn-synthetic',
                'knn_knn',
                'ehrmamba'
            ], axis=1, errors='ignore')
        except:
            pass
        dfs[task] = df_pivot
    return dfs

# Example usage:
dfs = prepare_elo_dataframe('/home/mpham/workspace/source/ehrpfn/outputs/results_l.csv')
global_elo_scores = {}

for task, df_task in dfs.items():
    if df_task.empty:
        continue
    elo_scores = calculate_average_elo_scores(df_task)
    for method, score in elo_scores.items():
        if method not in global_elo_scores:
            global_elo_scores[method] = []
        global_elo_scores[method].append(score)

# Calculate global average ELO scores
global_avg_elo_scores = {method: np.mean(scores) for method, scores in global_elo_scores.items()}
# Usage:
plot_elo_scores_with_std(global_elo_scores, path='/home/mpham/workspace/source/ehrpfn/outputs/visualization/performance/elo_l.pdf')