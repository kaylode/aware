import hydra
from omegaconf import DictConfig, OmegaConf
import os
import wandb
import time
from ehrpfn.datasets import get_mimic_dataset_v2
from ehrpfn.representation import SDAETransformer
from ehrpfn.representation.evaluate import evaluate_embeddings
from ehrpfn.utils.logger import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')


def logging_callback(
        epoch,
        lr,
        loss_value,
        validation_loss_value,
       knn_auc, mlp_auc
    ):

    LOGGER.text(f"Epoch {epoch}: lr={lr:.6f}, loss={loss_value:.6f}, validation_loss={validation_loss_value:.6f}, "
                 f"knn_auc={knn_auc:.6f}, mlp_auc={mlp_auc:.6f}")

    wandb.run.log({
        "lr": lr,
        "loss": loss_value,
        "validation_loss": validation_loss_value,
        "knn_auc": knn_auc,
        "mlp_auc": mlp_auc
    }, step=epoch)


@hydra.main(config_path='../../configs', config_name='sdae')
def main(cfg: DictConfig):
    LOGGER.text(OmegaConf.to_yaml(cfg))
    tags = ["MODEL:sdae", f"TASK:{cfg.training.task}"]

    save_path = cfg.training.save_path
    os.makedirs(save_path, exist_ok=True)
    save_name = "sdae"
    save_name += f"_{cfg.model.loss_function}"

    save_path = os.path.join(save_path, save_name)
    os.makedirs(save_path, exist_ok=True)

    tags.append(f"LOSS:{cfg.model.loss_function}")
    if cfg.model.gaussian_noise is not None:
        save_name += f"_noise{cfg.model.gaussian_noise}"
        tags.append(f"NOISE:{cfg.model.gaussian_noise}")

    if cfg.training.selection_method is not None:
        tags.append(f"SELECTION:{cfg.training.selection_method}")

    data = get_mimic_dataset_v2(
        data_csv=cfg.training.data_csv,
        train_label_path=cfg.training.train_label_path,
        test_label_path=cfg.training.test_label_path,
        feature_selection_path=cfg.training.feature_selection_path,
        selection_method=cfg.training.selection_method,
        process_file_path=None,
        trajectory_file_path=None
    )

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    LOGGER.text(f"X_train.shape: {X_train.shape}")
    LOGGER.text(f"X_test.shape: {X_test.shape}")


    wandb_kwargs = {
        "name": save_name,
        "entity": "kaylode",
        "project": "tabpfn-ehr",
        "reinit": True,
        "group": "sdae-mimicpipe",
        "job_type": "pretrain",
        "tags": tags,
        "resume": "allow",
        "dir": save_path,
    }
    wandb.init(**wandb_kwargs)
    wandb.run.config.update(OmegaConf.to_container(cfg, resolve=True))
    wandb.run.config.update({
        'study_name': save_name,
    })

    pipeline = SDAETransformer(
        dimensions=cfg.model.dimensions,
        pretrain_epochs=cfg.model.pretrain_epochs,
        finetune_epochs=cfg.model.finetune_epochs,
        verbose=True,
        batch_size=cfg.model.batch_size,
        cuda=cfg.model.cuda,
        valset=[X_test, y_test],
        noise_std=cfg.model.gaussian_noise,
        loss_function=cfg.model.loss_function,
        eval_freq=cfg.training.eval_freq,
        update_callback=logging_callback
    )

    running_time = time.time()
    pipeline = pipeline.fit(X_train, y_train)
    running_time = time.time() - running_time
    pipeline.save(f"{save_path}/{save_name}.pt")
    LOGGER.text(f"Model saved to {save_path}/{save_name}.pt")

    all_train = pipeline.transform(X_train)
    all_test = pipeline.transform(X_test)
   
    auc_score, mlp_auc = evaluate_embeddings(all_train, all_test, y_train, y_test)

    LOGGER.text(f"KNN AUC: {auc_score:.4f}")
    LOGGER.text(f"MLP AUC: {mlp_auc:.4f}")

    # Log the metrics to wandb
    wandb.run.log({
        "KNN AUC": auc_score,
        "MLP AUC": mlp_auc
    })
    wandb.run.log({"inference_time": running_time})
    wandb.finish()

    
if __name__ == "__main__":
    # import sys
    # import hydra
    # from omegaconf import OmegaConf
    main()
    # Default config
    # defaults = {
    #     "root": "/home/mpham/workspace/source/ehrpfn",
    #     "task": "Readmission",
    #     "loss_function": "huber",
    #     "gaussian_noise": None,
    #     "selection_method": None,
    #     "process_file_path": None,
    #     "trajectory_file_path": None,
    #     "pretrain_epochs": 1,
    #     "finetune_epochs": 256,
    #     "batch_size": 256,
    #     "cuda": True,
    # }
    # # Launch hydra with defaults if not using config files
    # hydra.main(config_path=None, config_name=None)(main)(OmegaConf.create(defaults))