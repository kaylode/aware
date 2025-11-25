import pandas as pd
import matplotlib.pyplot as plt
import json
import os

methods = ['knn', 'lightgbm', 'trompt', 'tabicl', 'tabpfn', 'tabdpt']
datasets = [
    'AIDS_Virus_Infection_Prediction',
    'COVID-19-Hospitals-Treatment-Plan',
    'Diabetes130US',
    'glioma_grading_clinical_and_mutation_features',
    'kidney',
    'Support2'
]
ratios = [1, 10, 20, 30, 50, 100]
# Collect results
missing = 0
results = []
for method in methods:
    for dataset_name in datasets:
        for ratio in ratios:
            score_json_file = f'/home/mpham/workspace/source/ehrpfn/outputs/all/results/openml_ehr_imbalance/{dataset_name}/{method}_{dataset_name}_{dataset_name}_retrieval:{method}_imbalance:{ratio}/score.json'
            if not os.path.exists(score_json_file):
                missing += 1
                continue
            try:
                with open(score_json_file, 'r') as f:
                    score = json.load(f)
                auc = score.get('auc', None)
                if auc is not None:
                    results.append({
                        'method': method,
                        'dataset': dataset_name,
                        'ratio': ratio,
                        'auc': auc
                    })
            except Exception as e:
                missing += 1
                continue

print(f'Missing files: {missing}')

# Convert to DataFrame
df = pd.DataFrame(results)

# Compute average AUC over datasets for each method and ratio
avg_auc = df.groupby(['method', 'ratio'])['auc'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
for method in methods:
    method_data = avg_auc[avg_auc['method'] == method]
    plt.plot(method_data['ratio'], method_data['auc'], marker='o', label=method)

plt.xlabel('Imbalance Ratio')
plt.ylabel('Average AUC')
plt.title('Model Performance (AUC) over Imbalance Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()