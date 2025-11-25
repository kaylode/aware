import os
import matplotlib as mpl
mpl.use("Agg")
import hydra
import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from omegaconf import DictConfig
import os.path as osp

from theseus.base.utilities.optuna_tuner import OptunaWrapper
from theseus.base.callbacks.optuna_callback import OptunaCallback
from source.pipelines import LightningPipeline


@hydra.main(version_base=None)
def main(opts: DictConfig):

    log_dir = osp.join(opts['global']['save_dir'], f"optuna", opts['global']['exp_name'])
    os.makedirs(log_dir, exist_ok=True)
    database = JournalStorage(
        JournalFileStorage(osp.join(log_dir, f"{opts['global']['exp_name']}.log"))
    )

    tuner = OptunaWrapper(
        storage=database,
        study_name=f"{opts['global']['exp_name']}",
        n_trials=10,
        direction="maximize",
        save_dir=log_dir,
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(),
        wandb_kwargs = {
            "entity": "kaylode",
            "project": "tabpfn-ehr",
            "reinit": True,
            "group": "tabpfn-ehr-optuna",
            "job_type": "optuna",
            "tags": [opts['global']['exp_name'], "optuna", opts['task_name']],
            "resume": "allow",
            "name": f"{opts['global']['exp_name']}",
            "dir": log_dir
        }
    )

    tuner.tune(
        config=opts,
        pipeline_class=LightningPipeline,
        optuna_callback=OptunaCallback,
        trial_user_attrs={
            "best_key": "weighted-f1",
            "model_name": opts['global']['exp_name'],
        },
    )

    leaderboard_df = tuner.leaderboard()
    leaderboard_df.to_json(
        osp.join(log_dir, "leaderboard.json"), orient="records"
    )

if __name__ == "__main__":
    main()