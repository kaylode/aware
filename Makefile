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
	bash scripts/experiment_retrieval.sh tabdpt mimiciv_hai shap 100
	bash scripts/experiment_retrieval.sh tabdpt eicu2 shap 100
	bash scripts/experiment_retrieval.sh knnpfn mimiciv_hai shap 100
	bash scripts/experiment_retrieval.sh knnpfn eicu2 shap 100
# 	bash scripts/experiment_retrieval.sh knn-simple mimiciv_hai shap 100
# 	bash scripts/experiment_retrieval.sh knn-simple eicu2 shap 100

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
	bash scripts/experiment_datasize.sh excelformer mimiciv_hai shap  null 500
	bash scripts/experiment_datasize.sh excelformer eicu2 shap  null 500
	bash scripts/experiment_datasize.sh trompt mimiciv_hai shap  null 500
	bash scripts/experiment_datasize.sh trompt eicu2 shap  null 500
	bash scripts/experiment_datasize.sh knn-simple mimiciv_hai shap  knn_l2 100
	bash scripts/experiment_datasize.sh knn-simple eicu2 shap  knn_l2 100
	bash scripts/experiment_datasize.sh tabdpt mimiciv_hai shap  knn_l2 100
	bash scripts/experiment_datasize.sh tabdpt eicu2 shap  knn_l2 100
	bash scripts/experiment_datasize.sh knnpfn mimiciv_hai shap  knn_l2 100
	bash scripts/experiment_datasize.sh knnpfn eicu2 shap  knn_l2 100
	bash scripts/experiment_datasize.sh tabpfn mimiciv_hai shap  null 100
	bash scripts/experiment_datasize.sh tabpfn eicu2 shap  null 100
	bash scripts/experiment_datasize.sh xgboost mimiciv_hai shap  null 500
	bash scripts/experiment_datasize.sh xgboost eicu2 shap  null 500
	bash scripts/experiment_datasize.sh modernNCA mimiciv_hai shap  null 500
	bash scripts/experiment_datasize.sh modernNCA eicu2 shap  null 500
	bash scripts/experiment_datasize.sh tabdpt mimiciv_hai shap  aware_snnl 100
	bash scripts/experiment_datasize.sh tabdpt eicu2 shap  aware_snnl 100
	bash scripts/experiment_datasize.sh knnpfn mimiciv_hai shap  aware_snnl 100
	bash scripts/experiment_datasize.sh knnpfn eicu2 shap  aware_snnl 100
# 	bash scripts/experiment_datasize.sh $(METHOD) openml_ehr null  $(RETRIEVAL) null
# 	bash scripts/experiment_datasize.sh $(METHOD) mimiciv shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_datasize.sh $(METHOD) eicu shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_datasize.sh $(METHOD) sjh_cpe_frame shap  $(RETRIEVAL) 100

exp_featuresize:
	bash scripts/experiment_featuresize.sh excelformer eicu2 shap  null
	bash scripts/experiment_featuresize.sh excelformer mimiciv_hai shap  null
	bash scripts/experiment_featuresize.sh trompt eicu2 shap  null
	bash scripts/experiment_featuresize.sh trompt mimiciv_hai shap  null
	bash scripts/experiment_featuresize.sh knn-simple eicu2 shap  knn_l2
	bash scripts/experiment_featuresize.sh knn-simple mimiciv_hai shap  knn_l2
	bash scripts/experiment_featuresize.sh tabdpt eicu2 shap  knn_l2
	bash scripts/experiment_featuresize.sh tabdpt mimiciv_hai shap  knn_l2
	bash scripts/experiment_featuresize.sh knnpfn eicu2 shap  knn_l2
	bash scripts/experiment_featuresize.sh knnpfn mimiciv_hai shap  knn_l2
	bash scripts/experiment_featuresize.sh xgboost eicu2 shap  null
	bash scripts/experiment_featuresize.sh xgboost mimiciv_hai shap  null
	bash scripts/experiment_featuresize.sh tabpfn eicu2 shap  null
	bash scripts/experiment_featuresize.sh tabpfn mimiciv_hai shap  null
	bash scripts/experiment_featuresize.sh modernNCA eicu2 shap  null
	bash scripts/experiment_featuresize.sh modernNCA mimiciv_hai shap  null
	bash scripts/experiment_featuresize.sh tabdpt eicu2 shap  aware_snnl
	bash scripts/experiment_featuresize.sh tabdpt mimiciv_hai shap  aware_snnl
	bash scripts/experiment_featuresize.sh knnpfn eicu2 shap  aware_snnl
	bash scripts/experiment_featuresize.sh knnpfn mimiciv_hai shap  aware_snnl
# 	bash scripts/experiment_featuresize.sh $(METHOD) eicu shap  $(RETRIEVAL)

exp_imbalance:
	bash scripts/experiment_imbalance_ratio.sh knn-simple eicu2 shap  knn_l2 500
	bash scripts/experiment_imbalance_ratio.sh knn-simple mimiciv_hai shap  knn_l2 500
	bash scripts/experiment_imbalance_ratio.sh tabdpt eicu2 shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt mimiciv_hai shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn eicu2 shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn mimiciv_hai shap  knn_l2 100
	bash scripts/experiment_imbalance_ratio.sh xgboost eicu2 shap  null 500
	bash scripts/experiment_imbalance_ratio.sh xgboost mimiciv_hai shap  null 500
	bash scripts/experiment_imbalance_ratio.sh tabpfn eicu2 shap  null 100
	bash scripts/experiment_imbalance_ratio.sh tabpfn mimiciv_hai shap  null 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt eicu2 shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh tabdpt mimiciv_hai shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn eicu2 shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh knnpfn mimiciv_hai shap  aware_snnl 100
	bash scripts/experiment_imbalance_ratio.sh excelformer eicu2 shap  null 500
	bash scripts/experiment_imbalance_ratio.sh excelformer mimiciv_hai shap  null 500
	bash scripts/experiment_imbalance_ratio.sh trompt eicu2 shap  null 500
	bash scripts/experiment_imbalance_ratio.sh trompt mimiciv_hai shap  null 500
	bash scripts/experiment_imbalance_ratio.sh modernNCA eicu2 shap  null 500
	bash scripts/experiment_imbalance_ratio.sh modernNCA mimiciv_hai shap  null 500

# 	bash scripts/experiment_imbalance_ratio.sh $(METHOD) eicu shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_imbalance_ratio.sh $(METHOD) mimiciv shap  $(RETRIEVAL) 100
# 	bash scripts/experiment_imbalance_ratio.sh $(METHOD) sjh_cpe_frame shap  $(RETRIEVAL) 100

benchmark_aware:
	bash scripts/inference.sh tabdpt $(DATASET) shap aware_snnl 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knnpfn $(DATASET) shap aware_snnl 100 $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null
	bash scripts/inference.sh knn-simple $(DATASET) null aware_snnl null $(CONTEXT_SIZE) $(INF_BATCH_SIZE) null


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


rebuttal:
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) knn_l2
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) attention
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) balance
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) minority_prioritized
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) inverse_frequency
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) siamese
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) supcon
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) unsupcon
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_nosup_full
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_attn_only
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_snnl_only
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_balance_only
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_ensemble_only
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_adapter_only
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_imb
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_no_attn
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_no_snnl
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_no_balance
# 	bash scripts/inference_ablation.sh $(METHOD) $(DATASET) aware_no_ensemble

# 	bash scripts/inference.sh knnpfn $(DATASET) null aware_imb null 1024 512 null
# 	bash scripts/inference.sh tabdpt $(DATASET) null aware_imb null 1024 512 null
# 	bash scripts/inference.sh knn-simple $(DATASET) null knn_l2 null 1024 512 null
# 	bash scripts/inference.sh knn-simple $(DATASET) null aware_imb null 1024 512 null
# 	bash scripts/inference.sh knnpfn $(DATASET) null knn_l2 null 1024 512 null
# 	bash scripts/inference.sh tabdpt $(DATASET) null knn_l2 null 1024 512 null


# 	bash scripts/inference.sh tabdpt $(DATASET) shap aware_imb 100 1024 512 null
# 	bash scripts/inference.sh tabdpt-aware $(DATASET) shap aware_imb 100 1024 128 null
	bash scripts/inference.sh knnpfn-aware $(DATASET) shap aware_imb 100 1024 128 null
# 	bash scripts/inference.sh knnpfnv1 $(DATASET) null knn_l2 null 1024 512 null
# 	bash scripts/inference.sh knnpfnv1 $(DATASET) null aware_imb null 1024 512 null
# 	bash scripts/inference.sh tabpfnv1 $(DATASET) null random null 1024 512 null
# 	bash scripts/inference.sh knnpfnv1 $(DATASET) shap aware_imb 100 1024 512 null
# 	bash scripts/inference.sh knnpfnv1 $(DATASET) shap knn_l2 100 1024 512 null
# 	bash scripts/inference.sh tabpfnv1 $(DATASET) shap random 100 1024 512 null
# 	bash scripts/inference.sh knn-simple $(DATASET) shap aware_imb 100 1024 512 null
# 	bash scripts/inference.sh knn-simple $(DATASET) shap knn_l2 100 1024 512 null


	bash scripts/inference.sh xgboost $(DATASET) null null null null null null
# 	bash scripts/inference.sh catboost $(DATASET) null null null null null null
# 	bash scripts/inference.sh lightgbm $(DATASET) null null null null null null
# 	bash scripts/inference.sh logistics $(DATASET) null null null null null null
# 	bash scripts/inference.sh mlp $(DATASET) null null null null null null
# 	bash scripts/inference.sh tabpfn $(DATASET) null null null 1024 512 null
# 	bash scripts/inference.sh knn-simple $(DATASET) null knn_l2 null 1024 512 null
# 	bash scripts/inference.sh knnpfn $(DATASET) null knn_l2 null 1024 512 null
# 	bash scripts/inference.sh tabdpt $(DATASET) null knn_l2 null 1024 512 null
# 	bash scripts/inference.sh resnet $(DATASET) null null null null null null
# 	bash scripts/inference.sh tabnet $(DATASET) null null null null null null
# 	bash scripts/inference.sh tabtransformer $(DATASET) null null null null null null
# 	bash scripts/inference.sh trompt $(DATASET) null null null null null null
	bash scripts/inference.sh excelformer $(DATASET) null null null null null null
# 	bash scripts/inference.sh tabr $(DATASET) null null null null null null
# 	bash scripts/inference.sh tabm $(DATASET) null null null null null null
# 	bash scripts/inference.sh modernNCA $(DATASET) null null 100 1024 512 null
