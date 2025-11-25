import pandas as pd
import matplotlib.pyplot as plt

imbalance_ratio = {
    "kidney": 1.125,
    "Diabetes130US": 4.830936307601996,
    "postoperative-patient-data": 2.590909090909091,
    "COVID-19-Hospitals-Treatment-Plan": 31.685848252344417,
    "Indian-Liver-Patient-Dataset": 2.5357142857142856,
    "Heart_Failure_Prediction": 2.1787584143605088,
    "AIDS_Virus_Infection_Prediction": 2.2184778492995076,
    "glioma_grading_clinical_and_mutation_features": 1.3817034700315458,
    "differentiated_thyroid_cancer_recurrence": 2.5463917525773194,
    "sepsis_survival_minimal_clinical_records": 12.573810825587753,
    "Support2": 2.1346595256312164,
    "Readmission": 6.184,
    "Length of Stay": 5.701,
    "Mortality": 8.936,
    "ICD SEPSIS": 7.451,
    "MORTALITY": 21.64,
    "READMISSION": 5.530,
}

# Read the wide-format CSV
result = pd.read_csv("averaged_ranks.csv")
result2 = pd.read_csv("averaged_ranks2.csv")

# Melt to long format: dataset_name, model, rank
result_long = result.melt(id_vars='task_name', var_name='model', value_name='rank')
result_long2 = result2.melt(id_vars='task_name', var_name='model', value_name='rank')

result_long = pd.concat([result_long, result_long2], axis=0, ignore_index=True)
# Map imbalance ratio
result_long['imbalance_ratio'] = result_long['task_name'].map(imbalance_ratio)


# Plot rank vs. imbalance ratio
# plt.figure(figsize=(10, 6))
# for model in result_long['model'].unique():
#     df_model = result_long[result_long['model'] == model]
#     plt.scatter(df_model['imbalance_ratio'], df_model['rank'], label=model, alpha=0.7)

# plt.xlabel('Imbalance Ratio')
# plt.ylabel('Model Rank')
# plt.title('Model Ranking vs. Imbalance Ratio')
# plt.gca().invert_yaxis()  # So rank 1 is at the top
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('ranking_vs_imbalance_ratio.pdf')


import numpy as np

# Bin imbalance ratios
bins = [0, 5, 10, 50]
labels = ['Low', 'Medium', 'High']
result_long['ratio_bin'] = pd.cut(result_long['imbalance_ratio'], bins=bins, labels=labels)

plt.figure(figsize=(10, 6))
for model in result_long['model'].unique():
    avg_rank = result_long.groupby('ratio_bin').apply(lambda x: x[x['model'] == model]['rank'].mean())
    plt.plot(labels, avg_rank, marker='o', label=model)

plt.xlabel('Imbalance Ratio Bin')
plt.ylabel('Average Model Rank')
plt.title('Average Model Rank by Imbalance Ratio Bin')
plt.gca().invert_yaxis()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('avg_rank_by_imbalance_bin.pdf')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ...existing code for imbalance_ratio, reading CSV, melting, binning...

# Calculate average rank for each model in each bin
avg_rank_bin = result_long.groupby(['ratio_bin', 'model'])['rank'].mean().unstack()

x = np.arange(len(labels))  # the label locations
width = 0.7 / len(avg_rank_bin.columns)  # width of the bars

plt.figure(figsize=(12, 6))
for i, model in enumerate(avg_rank_bin.columns):
    plt.bar(x + i * width, avg_rank_bin[model][labels], width, label=model)

plt.xlabel('Imbalance Ratio Bin')
plt.ylabel('Average Model Rank')
plt.title('Average Model Rank by Imbalance Ratio Bin')
plt.xticks(x + width * (len(avg_rank_bin.columns) - 1) / 2, labels)
plt.gca().invert_yaxis()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('bar_avg_rank_by_imbalance_bin.pdf')