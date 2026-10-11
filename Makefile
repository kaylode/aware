# Default values
METHOD ?= xgboost
DATASET ?= mimiciv
SELECTION ?= null
RETRIEVAL ?= null
NUM_FEATURES ?= null
CONTEXT_SIZE ?= 1024
INF_BATCH_SIZE ?= 512
SAMPLING ?= null

EICU_VERSION=eicu2
MIMIC_VERSION=mimiciv_hai

initialize:
	bash scripts/initialize.sh

inference:
	bash scripts/inference.sh $(METHOD) $(DATASET) $(SELECTION)  $(RETRIEVAL) $(NUM_FEATURES) $(CONTEXT_SIZE) $(INF_BATCH_SIZE) $(SAMPLING)

exp_retrieval:
	bash scripts/experiment_retrieval.sh tabdpt $(MIMIC_VERSION) shap 100
	bash scripts/experiment_retrieval.sh tabdpt $(EICU_VERSION) shap 100
	bash scripts/experiment_retrieval.sh knnpfnv1 $(MIMIC_VERSION) shap 100
	bash scripts/experiment_retrieval.sh knnpfnv1 $(EICU_VERSION) shap 100
	bash scripts/experiment_retrieval.sh tabdpt-aware $(MIMIC_VERSION) shap 100
	bash scripts/experiment_retrieval.sh tabdpt-aware $(EICU_VERSION) shap 100
	bash scripts/experiment_retrieval.sh knnpfn-aware $(MIMIC_VERSION) shap 100
	bash scripts/experiment_retrieval.sh knnpfn-aware $(EICU_VERSION) shap 100
	bash scripts/experiment_retrieval.sh knn-simple $(MIMIC_VERSION) shap 100
	bash scripts/experiment_retrieval.sh knn-simple $(EICU_VERSION) shap 100

exp_datasize:
	bash scripts/experiment_datasize.sh excelformer $(MIMIC_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh excelformer $(EICU_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh trompt $(MIMIC_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh trompt $(EICU_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh knn-simple $(MIMIC_VERSION) shap  knn_l2 100
	bash scripts/experiment_datasize.sh knn-simple $(EICU_VERSION) shap  knn_l2 100
	bash scripts/experiment_datasize.sh tabdpt $(MIMIC_VERSION) shap  knn_l2 100
	bash scripts/experiment_datasize.sh tabdpt $(EICU_VERSION) shap  knn_l2 100
	bash scripts/experiment_datasize.sh knnpfnv1 $(MIMIC_VERSION) shap  knn_l2 100
	bash scripts/experiment_datasize.sh knnpfnv1 $(EICU_VERSION) shap  knn_l2 100
	bash scripts/experiment_datasize.sh tabpfnv1 $(MIMIC_VERSION) shap  random 100
	bash scripts/experiment_datasize.sh tabpfnv1 $(EICU_VERSION) shap  random 100
	bash scripts/experiment_datasize.sh xgboost $(MIMIC_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh xgboost $(EICU_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh modernNCA $(MIMIC_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh modernNCA $(EICU_VERSION) shap  null 500
	bash scripts/experiment_datasize.sh tabdpt $(MIMIC_VERSION) shap  aware_imb 100
	bash scripts/experiment_datasize.sh tabdpt $(EICU_VERSION) shap  aware_imb 100
	bash scripts/experiment_datasize.sh knnpfn-aware $(MIMIC_VERSION) shap  aware_imb 100
	bash scripts/experiment_datasize.sh knnpfn-aware $(EICU_VERSION) shap  aware_imb 100

exp_featuresize:
	bash scripts/experiment_featuresize.sh excelformer $(EICU_VERSION) shap  null
	bash scripts/experiment_featuresize.sh excelformer $(MIMIC_VERSION) shap  null
	bash scripts/experiment_featuresize.sh trompt $(EICU_VERSION) shap  null
	bash scripts/experiment_featuresize.sh trompt $(MIMIC_VERSION) shap  null
	bash scripts/experiment_featuresize.sh knn-simple $(EICU_VERSION) shap  knn_l2
	bash scripts/experiment_featuresize.sh knn-simple $(MIMIC_VERSION) shap  knn_l2
	bash scripts/experiment_featuresize.sh tabdpt $(EICU_VERSION) shap  knn_l2
	bash scripts/experiment_featuresize.sh tabdpt $(MIMIC_VERSION) shap  knn_l2
	bash scripts/experiment_featuresize.sh knnpfnv1 $(EICU_VERSION) shap  knn_l2
	bash scripts/experiment_featuresize.sh knnpfnv1 $(MIMIC_VERSION) shap  knn_l2
	bash scripts/experiment_featuresize.sh xgboost $(EICU_VERSION) shap  null
	bash scripts/experiment_featuresize.sh xgboost $(MIMIC_VERSION) shap  null
	bash scripts/experiment_featuresize.sh tabpfnv1 $(EICU_VERSION) shap  random
	bash scripts/experiment_featuresize.sh tabpfnv1 $(MIMIC_VERSION) shap  random
	bash scripts/experiment_featuresize.sh modernNCA $(EICU_VERSION) shap  null
	bash scripts/experiment_featuresize.sh modernNCA $(MIMIC_VERSION) shap  null
	bash scripts/experiment_featuresize.sh tabdpt $(EICU_VERSION) shap  aware_imb
	bash scripts/experiment_featuresize.sh tabdpt $(MIMIC_VERSION) shap  aware_imb
	bash scripts/experiment_featuresize.sh knnpfn-aware $(EICU_VERSION) shap  aware_imb
	bash scripts/experiment_featuresize.sh knnpfn-aware $(MIMIC_VERSION) shap  aware_imb

exp_imbalance:
	bash scripts/experiment_imbalance_ratio.sh knn-simple $(EICU_VERSION) shap  knn_l2 500
	bash scripts/experiment_imbalance_ratio.sh knn-simple $(MIMIC_VERSION) shap  knn_l2 500
	bash scripts/experiment_imbalance_ratio.sh tabdpt $(EICU_VERSION) shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt $(MIMIC_VERSION) shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh knnpfnv1 $(EICU_VERSION) shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh knnpfnv1 $(MIMIC_VERSION) shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh xgboost $(EICU_VERSION) shap  null 500
	bash scripts/experiment_imbalance_ratio.sh xgboost $(MIMIC_VERSION) shap  null 500
	bash scripts/experiment_imbalance_ratio.sh tabpfnv1 $(EICU_VERSION) shap  random 100
	bash scripts/experiment_imbalance_ratio.sh tabpfnv1 $(MIMIC_VERSION) shap  random 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt-aware $(EICU_VERSION) shap  aware_imb 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt-aware $(MIMIC_VERSION) shap  aware_imb 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn-aware $(EICU_VERSION) shap  aware_imb 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn-aware $(MIMIC_VERSION) shap  aware_imb 100
	bash scripts/experiment_imbalance_ratio.sh excelformer $(EICU_VERSION) shap  null 500
	bash scripts/experiment_imbalance_ratio.sh excelformer $(MIMIC_VERSION) shap  null 500
	bash scripts/experiment_imbalance_ratio.sh trompt $(EICU_VERSION) shap  null 500
	bash scripts/experiment_imbalance_ratio.sh trompt $(MIMIC_VERSION) shap  null 500
	bash scripts/experiment_imbalance_ratio.sh modernNCA $(EICU_VERSION) shap  null 500
	bash scripts/experiment_imbalance_ratio.sh modernNCA $(MIMIC_VERSION) shap  null 500

benchmark:
	bash scripts/inference.sh xgboost $(DATASET) null  null null null null null
	bash scripts/inference.sh logistics $(DATASET) null null null null null null
	bash scripts/select_features.sh xgboost $(DATASET)
	bash scripts/inference.sh catboost $(DATASET) shap null 500 null null null
	bash scripts/inference.sh xgboost $(DATASET) shap null 500 null null null
	bash scripts/inference.sh lightgbm $(DATASET) shap null 500 null null null
	bash scripts/inference.sh logistics $(DATASET) shap null 500 null null null
	bash scripts/inference.sh mlp $(DATASET) shap null 500 null null null
	bash scripts/inference.sh knn-simple $(DATASET) shap knn_l2 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knn-simple $(DATASET) shap aware_imb 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knnpfnv1 $(DATASET) shap knn_l2 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabdpt $(DATASET) shap knn_l2 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabpfnv1 $(DATASET) shap random 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knnpfn-aware $(DATASET) shap aware_imb 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabdpt-aware $(DATASET) shap aware_imb 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh resnet $(DATASET) shap null 500 null null null
	bash scripts/inference.sh tabnet $(DATASET) shap null 500 null null null
	bash scripts/inference.sh tabtransformer $(DATASET) shap null 500 null null null
	bash scripts/inference.sh trompt $(DATASET) shap null 500 null null null
	bash scripts/inference.sh excelformer $(DATASET) shap null 500 null null null
	bash scripts/inference.sh tabm $(DATASET) shap null 500 null null null
	bash scripts/inference.sh modernNCA $(DATASET) shap null 500 null null null

selection:
	bash scripts/select_features.sh $(METHOD) $(DATASET)