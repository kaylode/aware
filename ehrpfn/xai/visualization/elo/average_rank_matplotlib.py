import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

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
    "G": ["TabPFN",  "TabICL", 'BETA'], # PFN-based
    "J": ['TabDPT', 'kNNPFN'], # Retrieval PFN-based
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
    "J": "#26A69A",  # Soft Indigo
}

type_comments = {
    "A": "Retrieval-based",
    "B": "Classical ML",
    "C": "GBMS",
    "D": "Deep Learning",
    "G": "PFN-based",
    "H": "Transformer-based",
    "I": "Sequential / EHR-specific",
    "J": "Retrieval PFN-based"
}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['Arial']


def get_metric_for_task(task_name, is_classification):
    # You can customize this logic as needed
    if is_classification:
        if task_name in ['Diabetes130US', 'COVID-19-Hospitals-Treatment-Plan']:
            return 'f1'
        else:
            return 'auc'
    else:
        return 'mae'


def prepare_rank_dataframe(csv_path):
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
        df_task = df_task[df_task[metric].notnull()]
        df_pivot = df_task.pivot_table(index='task_name', columns='Name', values=metric, aggfunc='mean')
        if not is_classification and metric == 'mae':
            df_pivot = -df_pivot
        df_pivot = df_pivot.rename(columns=names_dict)
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


def calculate_ranks_for_each_dataset(df):
    # For each dataset, rank the methods (higher is better)
    ranks_df = df.rank(axis=1, ascending=False)
    ranks_df = ranks_df.fillna(ranks_df.mean())
    return ranks_df


def plot_average_ranks(ranks_df, path):
    average_ranks = ranks_df.mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    colors = []
    for model in average_ranks.index:
        for key, value in method_types.items():
            if model in value:
                colors.append(colors_type[key])
                break
    bars = plt.barh(average_ranks.index, average_ranks.values, color=colors, edgecolor='black', alpha=0.5)
    for bar in bars:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.2f}', va='center', fontsize=10)
    plt.xlim(5, 32)
    plt.xlabel("Average Rank", fontsize=20)
    plt.ylabel("Methods", fontsize=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)


def plot_global_average_ranks(global_ranks, global_stds, path):
    methods = list(global_ranks.keys())
    means = np.array([global_ranks[m] for m in methods])
    stds = np.array([global_stds[m] for m in methods])
    sorted_idx = np.argsort(means)[::-1]
    methods = [methods[i] for i in sorted_idx]
    means = means[sorted_idx]
    stds = stds[sorted_idx]
    plt.figure(figsize=(10, 8))
    colors = []
    for model in methods:
        for key, value in method_types.items():
            if model in value:
                colors.append(colors_type[key])
                break
        else:
            raise ValueError(f"Missing color for {model}")
    bars = plt.barh(methods, means, xerr=stds, color=colors, edgecolor='black', alpha=0.5, capsize=5)
    for index, value in enumerate(means):
        plt.text(value + stds[index] + 0.05, index, f"{value:.2f}±{stds[index]:.2f}", va='center', fontsize=10)
    plt.xlabel("Global Average Rank", fontsize=20)
    plt.ylabel("Methods", fontsize=20)
    plt.xlim(0, 32)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    # Legend
    
    legend_elements = []
    for key in method_types.keys():
        legend_elements.append(Patch(facecolor=colors_type[key], edgecolor='black', label=f"{type_comments.get(key, key)}"))
    plt.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.95, 0.95),
        ncol=1,
        fontsize=11,
        title="Method Type",
        frameon=True
    )
    plt.tight_layout()
    plt.savefig(path)


# Usage example:
dfs = prepare_rank_dataframe('/home/mpham/workspace/source/ehrpfn/outputs/results_sm.csv')



# --- Global ranking calculation ---
all_ranks = []

for task, df_task in dfs.items():
    if df_task.empty:
        continue
    ranks_df = calculate_ranks_for_each_dataset(df_task)
    all_ranks.append(ranks_df)

# Concatenate all ranks
concat_ranks = pd.concat(all_ranks)
stored_df = pd.concat([i.reset_index() for i in all_ranks])
stored_df.to_csv('/home/mpham/workspace/source/ehrpfn/outputs/visualization/performance/averaged_ranks_sm.csv', index=False)
global_ranks = concat_ranks.mean().to_dict()
global_stds = concat_ranks.std().to_dict()
plot_global_average_ranks(global_ranks, global_stds, path='/home/mpham/workspace/source/ehrpfn/outputs/visualization/performance/rank_sm.pdf')