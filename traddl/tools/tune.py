import os
import matplotlib as mpl
mpl.use("Agg")
import hydra
import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from omegaconf import DictConfig

from theseus.base.utilities.optuna_tuner import OptunaWrapper
from theseus.base.callbacks.optuna_callback import OptunaCallback
from source.pipelines import LightningPipeline


@hydra.main(version_base=None)
def main(opts: DictConfig):
    
    os.makedirs("runs/optuna", exist_ok=True)
    database = JournalStorage(
        JournalFileStorage("runs/optuna/corebehrt_tune.log")
    )

    tuner = OptunaWrapper(
        storage=database,
        study_name="corebehrt_tune",
        n_trials=2,
        direction="maximize",
        save_dir=opts['global']['save_dir'],
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(),
        wandb_kwargs = {
            "entity": "kaylode",
            "project": "tabpfn-ehr",
            "reinit": True,
            "group": "tabpfn-ehr-optuna",
            "job_type": "optuna",
            "tags": [opts["model"]["name"], "optuna", opts['task_name']],
            "resume": "allow",
            "name": f"{opts["model"]["name"]}",
            "dir": opts['global']['save_dir']
        }
    )

    tuner.tune(
        config=opts,
        pipeline_class=LightningPipeline,
        optuna_callback=OptunaCallback,
        trial_user_attrs={
            "best_key": "weighted-f1",
            "model_name": opts["model"]["name"],
        },
    )

    leaderboard_df = tuner.leaderboard()
    os.makedirs("runs/optuna/overview", exist_ok=True)
    leaderboard_df.to_json(
        "runs/optuna/overview/leaderboard.json", orient="records"
    )

if __name__ == "__main__":
    main()