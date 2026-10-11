# Zero-Shot Efficiency of Prior-Fitted Networks on Electronic Health Records



## Core Idea:

We’re investigating how well tabular Prior-Fitted Networks (PFNs) and their variants—originally designed for general tabular data—can handle Electronic Health Records (EHRs), which are much more complex and heterogeneous. The project is empirical and diagnostic, not proposing a new architecture, but rather systematically benchmarking and analyzing their limitations and behaviors under realistic EHR conditions.

Contributons:
- Retrieval + Rarity: Improved hybrid retrieval with dense and retrieval for EHR --> build up context 
- Noisy ehr: Task-aware feature selection (currently using LR to find coefficients for features) 
- Longitudinal: Temporal feature aggregation (statistics of time series features engineering) --> lets use tfresh; should limit time series features to a known set of important features for some tasks
- Generalization: Pretrain feature encoder on synthetic EHRs --> we first pretrain its input embedding layers to capture the statistical structure of EHRs. --> then TabPFN perform bayesian inference over.

Assumptions made:
- The distribution of input features in the synthetic EHRs is similar enough to that in real or downstream EHR tasks.
- The synthetic data preserves relationships between features, not just marginal distributions.
- There exists a shared representation that’s useful across all EHR tasks (e.g., mortality, readmission, diagnosis prediction).
- The pretrained encoder provides an implicit prior over what EHR inputs look like.

## 🎯 Main Objectives (Research Questions)

We structured the paper and experiments around four domain-specific challenges for PFNs in EHRs:

#### Heterogeneity
> Can PFNs and other tabular foundation models handle mixed feature types (ICD codes, vitals, demographics, labs) and diverse dataset structures?

#### Rarity
> How do PFNs perform on rare clinical outcomes with high class imbalance, compared to both classical ML and deep tabular models?

#### Temporality
> Can PFNs (designed for static tables) still benefit from temporal feature engineering—simple longitudinal summaries compatible with tabular inputs?

#### Retrieval / RAICL (Retrieval-Augmented In-Context Learning) —
> Does retrieval help or hurt PFN performance on EHRs?
> When is retrieval useful vs. noisy?
> What retrieval mechanisms (raw kNN, weighted retrieval, learned metrics, ICD embeddings, etc.) work best?

## 🧩 Scope of the Study

#### Datasets:

- Large, longitudinal EHRs:
    - MIMIC-IV — mortality, readmission, length-of-stay, rare infections
    - eICU — mortality and readmission (cross-hospital generalization)
- Small–medium EHR datasets:
    - 20 datasets from OpenML EHR Collection (for diverse, heterogeneous benchmarking)

#### Models Benchmarked:

- Classical baselines: Logistic Regression, kNN, XGBoost
- General deep tabular models: TabNet, TabTransformer, ResNet, TabM, TrompT, ExcelFormer
- Retrieval-augmented / ICL-based: TabPFN, TabDPT, TabR, ModernNCA, TabICL
- EHR-specific sequential baselines (optional): CoreBEHRT, EHRMamba (for temporal comparison)

#### Evaluation metrics:

- AUROC, AUPRC, F1, MAE (for regression), precision@k (retrieval quality), ECE (calibration).