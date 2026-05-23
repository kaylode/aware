from typing import *
import os
import os.path as osp
import numpy as np
import hydra
import json
import wandb
from omegaconf import DictConfig

from ehrpfn.data import get_data, prepare_data
from ehrpfn.models import get_predictions
from ehrpfn.utils import FileLogger, LoggerObserver, seed_everything, setup_inference
from ehrpfn.metrics import run_binary_classification_metrics, run_regression_metrics


LOGGER = LoggerObserver.getLogger("main")


def inference(args: DictConfig, data):

	result_folder, study_name = setup_inference(args, data)
	LOGGER.subscribe(FileLogger(__name__, result_folder, debug=True))

	if os.path.exists(osp.join(result_folder, 'score.json')):
		score = json.load(open(osp.join(result_folder, 'score.json'), 'r'))

		is_classification = data.get('is_classification', False)
		should_skip = True
		if is_classification:
			auc = score.get('auc')
			auprc = score.get('auprc')
			if auc is None or (isinstance(auc, float) and np.isnan(auc)) or \
			   auprc is None or (isinstance(auprc, float) and np.isnan(auprc)):
				should_skip = False

		if should_skip:
			LOGGER.text(f"Score already exists for {study_name} in {result_folder}. Skipping inference.", level=LoggerObserver.INFO)
			LOGGER.text(f"Score: {score}", level=LoggerObserver.INFO)
			return
		else:
			LOGGER.text(f"Existing score for {study_name} in {result_folder} has NaN/None for AUC or AUPRC. Rerunning inference.", level=LoggerObserver.INFO)
	
	predictions, training_time, running_time = get_predictions(args, data, {
		"result_folder": result_folder,
		"study_name": study_name
	})

	y_test = data['y_test']
	is_classification = data['is_classification']
	if is_classification:
		score = run_binary_classification_metrics(
			predictions, 
			y_test,
			plot_save_path=osp.join(result_folder, 'prediction_distribution.pdf')
		)
	else:
		score = run_regression_metrics(
			predictions, 
			y_test,
		)

	score.update({"inference_time": running_time, 'training_time': training_time})
	
	# Log the metrics to wandb
	if args.wandb.use_wandb:
		wandb.run.log(score)
		wandb.finish()

	# Dump score into JSON
	with open(osp.join(result_folder, 'score.json'), 'w') as f:
		json.dump(score, f, indent=4)

	LOGGER.text(f"Score: {score}", level=LoggerObserver.INFO)


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def run(args: DictConfig):

	seed_everything(args.inference.seed)

	data = get_data(args)

	if isinstance(data, list):
		for d in data:
			# try:
			data = prepare_data(d, args)
			inference(args, data)
			# except Exception as e:
				# LOGGER.text(f"Error occurred while processing {d['data_name']}: {e}", level=LoggerObserver.ERROR)
	else:
		# try:
		data = prepare_data(data, args)
		inference(args, data)
		# except Exception as e:
			# LOGGER.text(f"Error occurred during inference: {e}", level=LoggerObserver.ERROR)
if __name__ == "__main__":
	run()