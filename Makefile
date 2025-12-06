# Default values
METHOD ?= xgboost
DATASET ?= mimiciv

inference:
	bash scripts/inference.sh $(METHOD) $(DATASET)

selection:
	bash scripts/select_features.sh $(METHOD) $(DATASET)