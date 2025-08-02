# TO-DO LIST

🛠️ Data Preparation

✅ [ ] Extract MIMIC-IV cohorts

- Filter for rare infections (SEPSIS, SSI, CLABSI, CAUTI, VAP).
- Apply patient-wise train-test split, k-fold CV setup.
- Extract:
    - Demographics (Age, Gender, Ethnicity)
    - Labs
    - Vitals
    - Medications / Diagnosis codes / Procedures
    - Ward transitions


[ ] Featue selection
- Benchmark SHAP values, xgboost and random forest

✅ [ ] Feature Engineering

- Aggregate time series (min, mean, max, threshold indicators, statistics).
- Select top infection-relevant features fitting TabPFN token limits.
    - Use SHAP values / feature selection methods
- Label assignment and balancing checks (positive/negative distribution).

🛠️ 2️⃣ Process Mining for Pathway Variants

✅ [ ] Build event logs:
- Lab/vital threshold crossings (e.g., WBC > 12, lactate > 2)
- Medication starts (e.g., first antibiotic, vasopressor initiation)
- Ward transitions (ED admission/discharge, ICU admission/discharge)

✅ [ ] Apply process mining + clustering:
- Discover X variants (decide X based on Silhouette/Gap statistic).
- Assign pathway variant labels to each patient for retrieval anchoring.

🛠️ 3️⃣ Retrieval Module

✅ [ ] Implement retrieval pipeline:

- For each test patient, retrieve k most similar patients from training set based on:
    - Demographics
    - Key labs/vitals
    - Pathway variant

- Experiment with different k values (e.g., 3, 5, 10) for ablation.

🛠️ 4️⃣ TabPFN Integration

✅ [ ] Format each retrieved patient example into TabPFN prompt format.
✅ [ ] Concatenate retrieved examples + test patient to form ICL prompt.
✅ [ ] Ensure compliance with TabPFN’s samples & features limit.
✅ [ ] Implement lightweight linear adapters (Mixture of In-Context Prompters) inside TabPFN
✅ [ ] Continual Learning on MIMIC-IV.

🛠️ 6️⃣ Baselines

✅ [ ] Train:

- Vanilla TabPFN, MixturePFN, LocalPFN
- LightGBM / XGBoost
- TabPFN + RAG-ICL
- TabPFN + RAG-ICL + Adapters

✅ [ ] Metrics:

- AUROC, AUPRC
- Calibration metrics (Brier score, ECE)


🛠️ 8️⃣ Ablation Studies

✅ [ ] Evaluate:

- Retrieval with vs. without pathway variants.
- Different k retrieval sizes.
- With vs. without adapters.


🛠️ 10️⃣ Visualization Scripts

✅ [ ] Generate:
- Embedding Projection of patient, categorized by process variants 
- Distribution of classes (vs. process variants)
- Process mining variant Sankey/cluster plots.
- AUROC/AUPRC bar/line plots.
- Calibration curves.
- Inference time bar plots.
- (Optional) Patient trajectory timelines for interpretability visuals.

🛠️ Stretch Tasks

✅ [ ] Test on external cohort (e.g., eICU) if available for generalization.
✅ [ ] Interpretability analysis: feature importance within retrieved examples.

