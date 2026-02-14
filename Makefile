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
	bash scripts/experiment_retrieval.sh knn-simple mimiciv_hai shap 100
	bash scripts/experiment_retrieval.sh tabdpt mimiciv_hai shap 100
	bash scripts/experiment_retrieval.sh knnpfn mimiciv_hai shap 100
	bash scripts/experiment_retrieval.sh knn-simple eicu2 shap 100
	bash scripts/experiment_retrieval.sh tabdpt eicu2 shap 100
	bash scripts/experiment_retrieval.sh knnpfn eicu2 shap 100

exp_sampling:
# 	bash scripts/experiment_sampling.sh tabdpt mimiciv_hai shap 100
# 	bash scripts/experiment_sampling.sh knnpfn mimiciv_hai shap 100
# 	bash scripts/experiment_sampling.sh knn-simple eicu2 shap 100
	bash scripts/experiment_sampling.sh tabpfn eicu2 shap 100
	bash scripts/experiment_sampling.sh tabicl eicu2 shap 100
# 	bash scripts/experiment_sampling.sh tabdpt eicu2 shap 100
# 	bash scripts/experiment_sampling.sh knnpfn eicu2 shap 100
# 	bash scripts/experiment_sampling.sh knn-simple mimiciv_hai shap 100
	bash scripts/experiment_sampling.sh tabpfn mimiciv_hai shap 100
	bash scripts/experiment_sampling.sh tabicl mimiciv_hai shap 100

exp_datasize:
	bash scripts/experiment_datasize.sh $(METHOD) openml_ehr null  $(RETRIEVAL) null
	bash scripts/experiment_datasize.sh $(METHOD) mimiciv shap  $(RETRIEVAL) 100
	bash scripts/experiment_datasize.sh $(METHOD) eicu shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_datasize.sh $(METHOD) sjh_cpe_frame shap  $(RETRIEVAL) 100

exp_featuresize:
	bash scripts/experiment_featuresize.sh knnpfn mimiciv shap aware_snnl
	bash scripts/experiment_featuresize.sh knnpfn mimiciv shap aware_supcon
# 	bash scripts/experiment_featuresize.sh tabdpt eicu shap aware_supcon
# 	bash scripts/experiment_featuresize.sh tabdpt mimiciv shap aware_snnl
# 	bash scripts/experiment_featuresize.sh tabdpt mimiciv shap aware_supcon
# 	bash scripts/experiment_featuresize.sh knnpfn eicu shap aware_snnl
# 	bash scripts/experiment_featuresize.sh knnpfn eicu shap aware_supcon
# 	bash scripts/experiment_featuresize.sh tabdpt eicu shap aware_snnl
# 	bash scripts/experiment_featuresize.sh $(METHOD) eicu shap  $(RETRIEVAL)
# 	bash scripts/experiment_featuresize.sh $(METHOD) mimiciv shap  $(RETRIEVAL)

exp_imbalance:
	bash scripts/experiment_imbalance_ratio.sh tabdpt eicu shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt mimiciv shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt eicu shap  aware_supcon 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt mimiciv shap  aware_supcon 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn eicu shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn mimiciv shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn eicu shap  aware_supcon 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn mimiciv shap  aware_supcon 100
# 	bash scripts/experiment_imbalance_ratio.sh $(METHOD) eicu shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_imbalance_ratio.sh $(METHOD) mimiciv shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_imbalance_ratio.sh $(METHOD) sjh_cpe_frame shap  $(RETRIEVAL) 100

benchmark_aware:
	bash scripts/inference.sh knn-simple $(DATASET) null aware_snnl null $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knn-simple $(DATASET) null aware_supcon null $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabdpt $(DATASET) shap aware_snnl 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabdpt $(DATASET) shap aware_supcon 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knnpfn $(DATASET) shap aware_snnl 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knnpfn $(DATASET) shap aware_supcon 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null


benchmark:
	bash scripts/inference.sh xgboost $(DATASET) null  null null null null null
	bash scripts/inference.sh logistics $(DATASET) null null null null null null
	bash scripts/select_features.sh xgboost $(DATASET)
	bash scripts/inference.sh catboost $(DATASET) shap null 500 null null null
	bash scripts/inference.sh xgboost $(DATASET) shap null 500 null null null
	bash scripts/inference.sh lightgbm $(DATASET) shap null 500 null null null
	bash scripts/inference.sh logistics $(DATASET) shap null 500 null null null
	bash scripts/inference.sh mlp $(DATASET) shap null 500 null null null
	bash scripts/inference.sh tabpfn $(DATASET) shap null 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabicl $(DATASET) shap null 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knn-simple $(DATASET) null knn_l2 null $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knnpfn $(DATASET) shap knn_l2 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh tabdpt $(DATASET) shap knn_l2 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null

	bash scripts/inference.sh danets $(DATASET) shap null 500 null null null
	bash scripts/inference.sh resnet $(DATASET) shap null 500 null null null
	bash scripts/inference.sh tabnet $(DATASET) shap null 500 null null null
	bash scripts/inference.sh tabtransformer $(DATASET) shap null 500 null null null
	bash scripts/inference.sh trompt $(DATASET) shap null 500 null null null
	bash scripts/inference.sh excelformer $(DATASET) shap null 500 null null null
# 	bash scripts/inference.sh tabr $(DATASET) shap null 500 null null null
	bash scripts/inference.sh tabm $(DATASET) shap null 500 null null null
	bash scripts/inference.sh modernNCA $(DATASET) shap null 500 null null null
	bash scripts/inference.sh beta $(DATASET) shap null 500 null null null

selection:
	bash scripts/select_features.sh $(METHOD) $(DATASET)