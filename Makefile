# Default values
METHOD ?= xgboost
DATASET ?= mimiciv
SELECTION ?= null
RETRIEVAL ?= null
NUM_FEATURES ?= null
CONTEXT_SIZE ?= 1024
INF_BATCH_SIZE ?= 512
SAMPLING ?= null

inference:
	bash scripts/inference.sh $(METHOD) $(DATASET) $(SELECTION)  $(RETRIEVAL) $(NUM_FEATURES) $(CONTEXT_SIZE) $(INF_BATCH_SIZE) $(SAMPLING)

exp_retrieval:
# 	bash scripts/experiment_retrieval.sh $(METHOD) mimiciv shap 100
# 	bash scripts/experiment_retrieval.sh $(METHOD) eicu shap 100
	bash scripts/experiment_retrieval.sh $(METHOD) sjh_cpe_frame shap 100

exp_sampling:
	bash scripts/experiment_sampling.sh ${METHOD} sjh_cpe_frame shap 100

exp_datasize:
	bash scripts/experiment_datasize.sh $(METHOD) openml_ehr null  $(RETRIEVAL) null
	bash scripts/experiment_datasize.sh $(METHOD) mimiciv shap  $(RETRIEVAL) 100
	bash scripts/experiment_datasize.sh $(METHOD) eicu shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_datasize.sh $(METHOD) sjh_cpe_frame shap  $(RETRIEVAL) 100

exp_featuresize:
	bash scripts/experiment_featuresize.sh $(METHOD) eicu shap  $(RETRIEVAL)
	bash scripts/experiment_featuresize.sh $(METHOD) mimiciv shap  $(RETRIEVAL)
# 	bash scripts/experiment_featuresize.sh $(METHOD) sjh_cpe_frame shap  $(RETRIEVAL)

exp_imbalance:
	bash scripts/experiment_imbalance_ratio.sh $(METHOD) eicu shap  $(RETRIEVAL) 100
	bash scripts/experiment_imbalance_ratio.sh $(METHOD) mimiciv shap  $(RETRIEVAL) 100
	bash scripts/experiment_imbalance_ratio.sh $(METHOD) sjh_cpe_frame shap  $(RETRIEVAL) 100

benchmark:
# 	bash scripts/inference.sh xgboost $(DATASET) null  null null null null null
# 	bash scripts/select_features.sh xgboost $(DATASET)
	bash scripts/inference.sh catboost $(DATASET) null null null null null null
	bash scripts/inference.sh lightgbm $(DATASET) null null null null null null
	bash scripts/inference.sh logistics $(DATASET) null null null null null null
	bash scripts/inference.sh mlp $(DATASET) null null null null null null
	bash scripts/inference.sh knn-simple $(DATASET) null knn_l2 null $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabpfn $(DATASET) shap null 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabicl $(DATASET) shap null 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knnpfn $(DATASET) shap knn_l2 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabdpt $(DATASET) shap knn_l2 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null

selection:
	bash scripts/select_features.sh $(METHOD) $(DATASET)