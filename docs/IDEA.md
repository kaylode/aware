🧠 Research Proposal: Adapting TabPFN for Predictive Modeling on Longitudinal Electronic Patient Records
# 1. Research Motivation and Gap
Recent breakthroughs in In-Context Learning (ICL) models like TabPFN and its variants have shown remarkable performance on small to medium-sized tabular datasets, outperforming traditional models like CatBoost and XGBoost, especially under data-scarce conditions. However, no existing work has yet explored adapting TabPFN to longitudinal Electronic Patient Record (EPR) data, which are characterized by:

- High temporal complexity (multiple time-stamped visits per patient)
- High dimensionality (thousands of clinical features)
- High heterogeneity and missingness

> This presents a unique opportunity to extend TabPFN's tabular strengths into the longitudinal healthcare domain, potentially outperforming existing methods that lack the inductive biases encoded via SCM pretraining or the robustness of ICL approaches.

# 2. Research Objectives
- This study aims to develop, adapt, and evaluate TabPFN-based models for predictive tasks on longitudinal EPR datasets. Key objectives:
- Adapt TabPFN for temporally structured EPR data, using feature engineering and/or structural architectural modifications (e.g. TabPFN-TS, TuneTables, LoCalPFN).
- Evaluate the model’s performance against strong baselines (CatBoost, XGBoost, RNNs, SSMs like Mamba/Hydra, FTTransformer).
- Assess interpretability, bias/variance behavior, and scalability of the adapted models.

# 3. Methodological Framework
## 3.1 Data Processing and Representation

We propose to flatten longitudinal patient data into engineered tabular representations, drawing on ideas from TabPFN-TS and time-series feature extraction:

- Static Features: age, sex, baseline diagnosis
- Temporal Features: time since last visit, time to next visit, visit intervals
- Aggregated Histories: rolling means, standard deviations, min/max of labs/vitals
- Lagged Variables: include prior visit values explicitly

Optional: use Fourier transforms (as in TabPFN-TS) for capturing cyclic/seasonal patterns

## 3.2 Model Variants to Explore
We will experiment with multiple TabPFN-related architectures adapted to EPR:

| Variant | Adaptation Purpose |
|---------|-------------------|
| TabPFN-v2 | Base foundation, for initial benchmarking |
| TabPFN-ICD | Handles larger datasets via in-context distillation |
| TuneTables | Scalable prompt compression and fine-tuning |
| MixturePFN | Enables clustering-based "expert" routing for sub-cohort modeling |
| TabPFN-TS (modified) | For prognosis or regression over disease trajectories |
| TabICL | Column-row attention for high-dimensional, large-cohort EPR data |
| LoCalPFN | kNN-based local prompts to personalize predictions |
| Do-PFN | To explore causal predictions (e.g., treatment effect modeling) |

Additionally, we may explore ensembling strategies (TabPFN default) and BETA (bias/variance optimized model) for enhanced generalization.

## 3.3 Target Tasks
We will evaluate the proposed system on key clinical predictive tasks, such as:

- Hospital readmission risk (binary classification)
- Disease progression (regression)
- Multi-label diagnosis prediction (multiclass)
- Treatment effect estimation (causal inference using Do-PFN)

# 4. Evaluation Strategy

### Datasets
- MIMIC-IV: ICU longitudinal data (structured + temporal)
- CPRD or other national EPRs (if accessible)
- Synthetic EPR datasets for pretraining, mimicking SCM priors

### Baselines
- CatBoost, XGBoost (strong tree models)
- SAINT, FTTransformer, ResNet (deep tabular networks)
- RNN/LSTM and SSM-based models (Mamba, Hydra)
- TabNet

### Metrics
- Classification: Accuracy, AUC, F1-score
- Regression: RMSE, MAE, R²
- Time-series forecasting: MASE (as used in TabPFN-TS)
- Causal inference: ATE/CATE estimation error, PEHE
- Interpretability: SHAP, prompt-based insights
- Scalability: Runtime, memory, accuracy-time trade-offs

# 5. Anticipated Contributions
- First empirical study adapting TabPFN for longitudinal EPR data
- Comprehensive benchmarking of TabPFN variants in clinical settings
- Novel hybrid framework combining TabPFN's ICL with time-aware tabular representations
- Insights into interpretability, robustness, and causal reasoning in healthcare foundation models

# 6. Limitations and Risks
- Temporal representation may be insufficient without sequence modeling
- Preprocessing and feature engineering introduce domain bias
- Performance may degrade on extremely large or highly missing datasets without further distillation
- Interpretability of ICL remains challenging, though mitigated by SHAP and causal priors