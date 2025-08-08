import os
import os.path as osp
import random
import signal

import hydra
import torch.distributed as dist
import schedulefree
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ehrpfn.utils.logger import LoggerObserver
from ehrpfn.training.datasets import FullDataset, collate_fn
from ehrpfn.training.eval_full import FullEval
from ehrpfn.training.eval_health import FullHealthEval
from ehrpfn.training.eval_mimic import MIMICIVOutcomesEval
from ehrpfn.training.model import TabDPTModel
from ehrpfn.training.utils import (
    cleanup,
    compute_losses,
    get_combined_loss,
    init_dist,
    log_param_norms_wandb,
    seed_everything,
    signal_handler,
)

LOGGER = LoggerObserver.getLogger('main')


def save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DictConfig,
    stats: dict,
    path: str,
    name: str,
) -> None:
    """Save the model and optimizer state.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        config (DictConfig): The configuration for the experiment.
        stats (dict): The training statistics.
        path (str): The path to save the checkpoint.
        name (str): The name of the checkpoint file.

    Returns:
        None
    """
    ckpt = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "cfg": config,
        "stats": stats,
        'wandb_id': wandb.run.id if wandb.run else None,
    }
    torch.save(ckpt, f"{path}/{name}.ckpt")


def save_eval_callback_wandb(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vals: list,
    epoch: int,
    config: DictConfig,
    rank: int,
    stats: dict,
    global_step:int
) -> None:
    """Save evaluation checkpoints and log metrics.

    Args:
        model (nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        vals (list): A list of validation datasets.
        epoch (int): The current epoch number.
        config (DictConfig): The configuration for the training process.
        writer (SummaryWriter): The TensorBoard writer for logging.
        rank (int): The rank of the current process.
        stats (dict): A dictionary to store training statistics.

    Returns:
        None
    """
    # save latest model
    stats["epoch_in_training"] = epoch
    if rank == 0:
        save(model, optimizer, config, stats, config.exp_path, "latest")

    # eval and save best
    model.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()

    for val in vals:
        for name, metric in val.eval(model, context_length=config.training.eval_seq_len).items():
            if rank == 0:
                LOGGER.text(f"Epoch {epoch} | {name}: {metric}")
                # only save checkpoints using the validation metric
                if metric > stats.get(f"best_{name}", -float("inf")):
                    stats[f"best_{name}"] = metric
                    if name in config.logging.save_metrics:
                        save(model, optimizer, config, stats, config.exp_path, f"best_{name}")
                        LOGGER.text(f"Saved best {name} model at epoch {epoch} with score={metric}", level=LoggerObserver.SUCCESS)
                wandb.log({f"val/{name}/": metric}, step=global_step)

    # reset the model and optimizer to training mode
    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()


def set_experiment_from_config(config: DictConfig) -> tuple[nn.Module, torch.optim.Optimizer, dict]:
    """Set up the experiment based on the provided configuration.

    Args:
        config (DictConfig): The configuration for the experiment.

    Raises:
        Exception: If the reset policy is not recognized.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer, dict]: A tuple containing the model,
        optimizer, and a dictionary with training statistics.
    """

    load_state_from_saved = False
    stats = {"epoch_in_training": 0}

    seed_everything(config.seed)


    config.exp_path = osp.join(config.folder, config.exp_name)

    # Only rank 0 should modify the directory
    is_master = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

    wandb_id = wandb.util.generate_id()
    # directory setup and checkpoint handling
    if is_master:
        if os.path.exists(config.exp_path):
            LOGGER.text(f"Directory already exists: {config.exp_path}")
            if config.training.resume is not None:
                load_state_from_saved = True
                checkpoint = torch.load(f"{config.training.resume}/latest.ckpt", weights_only=False)
                model_state = checkpoint["model"]
                opt_state = checkpoint["opt"]
                stats = checkpoint["stats"]
                wandb_id = checkpoint.get("wandb_id", wandb_id)
                non_saved_num_epochs = config.training.num_epochs
                config.training.num_epochs = non_saved_num_epochs
                LOGGER.text("Continue training. Using saved config.")
            elif config.training.pretrained is not None:
                load_state_from_saved = True
                checkpoint = torch.load(config.training.pretrained, weights_only=False)
                model_state = checkpoint["model"]
                opt_state = None
            else:
                raise ValueError(
                    "Invalid checkpoint path."
                )
        else:
            os.makedirs(config.exp_path, exist_ok=True)
            LOGGER.text(f"Created directory: {config.exp_path}")

    # Synchronize all processes so that the directory is created before continuing
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    assert (
        config.training.batch_size % len(config.env.gpus) == 0
    ), "Batch size should be divisible by the number of GPUs"

    # adapt batch size for distributed training
    config.training.batch_size //= len(config.env.gpus)

    # save config
    OmegaConf.save(config=config, f=f"{config.exp_path}/config.yaml")

    LOGGER.text(f"Using {config.env.gpus} GPUs")

    # instantiate the model
    model = TabDPTModel(
        dropout=config.training.dropout,
        n_out=config.model.max_num_classes,
        nhead=config.model.nhead,
        nhid=config.model.nhid_factor * config.model.emsize,
        ninp=config.model.emsize,
        nlayers=config.model.nlayers,
        num_features=config.model.max_num_features,
    )

    # load the model state if resuming from a saved checkpoint
    # TODO : handle model_state definition in a more robust way
    if load_state_from_saved:
        model = TabDPTModel.load(model_state, config)
        del model_state

    # instantiate the optimizer
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        warmup_steps=1000,
        betas=(0.98, 0.999),
    )

    # load the optimizer state if resuming from a saved checkpoint
    # TODO : handle model_state definition in a more robust way
    if load_state_from_saved and opt_state is not None:
        optimizer.load_state_dict(opt_state)
        del opt_state

    return model, optimizer, stats, wandb_id


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(config: DictConfig):
    """Main function to run the training process.

    Args:
        config (DictConfig): The configuration for the training process.
    """

    # signal handling
    signal.signal(signal.SIGINT, signal_handler)

    # LOGGER.text config
    LOGGER.text(OmegaConf.to_yaml(config, resolve=True))

    # distributed training
    using_dist, rank, device = init_dist(config.env.device)

    model, optimizer, stats, wandb_id = set_experiment_from_config(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.text(f"{num_params/1e6:.{2}f} M parameters")
    model.to(device)

    # compile the model if specified in the config
    if config.training.compile:
        model = torch.compile(model)

    # using distributed data parallel if specified in the config
    if using_dist:
        LOGGER.text("Distributed training enabled.")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
        )

    # set validation datasets
    # TODO : make this configurable

    vals = [
        FullEval(
            device=device,
            max_feat=config.model.max_num_features,
            use_retrieval=config.data.eval_retrieval,
            data_folder=config.data.openml.input_folder,
        ),
        FullHealthEval(
            device=device,
            max_feat=config.model.max_num_features,
            use_retrieval=config.data.eval_retrieval,
            data_folder=config.data.openml.input_folder,
        ),
        MIMICIVOutcomesEval(
            device=device,
            max_feat=config.model.max_num_features,
            use_retrieval=config.data.eval_retrieval,
            data_path=config.data.mimiciv.val_folder,
            label_path=config.data.mimiciv.val_label_folder,
        ),
    ]

       

    # select the device type
    device_type = "cpu" if config.env.device == "cpu" else "cuda"

    # Use autocast for mixed precision training
    selected_autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    # Initialize the FullDataset for training
    LOGGER.text("Initializing Dataset...")
    dataset = FullDataset(
        device, config, 
    )

    # initialize the DataLoader for the dataset
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=config.env.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        persistent_workers=True,
    )

    # Create an iterator for the DataLoader
    iter_data_loader = iter(data_loader)


    tags = [f"FEAT:{config.model.max_num_features}"]

    if config.data.use_health_data:
        tags.append("PRETRAIN:HEALTH")
    else:
        tags.append("PRETRAIN:DEFAULT")

    if config.data.use_synthetic:
        tags.append("PRETRAIN:SYNTHETIC")

    if config.data.use_mimiciv:
        tags.append("PRETRAIN:MIMICIV")

    ## Init wandb logger
    wandb_kwargs = {
        "name": f"{config.exp_name}",
        "entity": "kaylode",
        "project": "tabpfn-ehr",
        "reinit": True,
        "group": "tabpfn-ehr-pretrain",
        "job_type": "test",
        "tags": tags,
        "resume": "allow",
        "dir": osp.join(config.exp_path),
    }

    if config.training.resume is not None:
        wandb_kwargs["resume"] = "allow"
        wandb_kwargs["id"] = wandb_id

    wandb.init(**wandb_kwargs)
    wandb.run.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)
    wandb.run.config.update({
        'study_name': config.exp_name,
        'data_name': "",
        'target_column': "",
        "MODEL": {
            "MODEL_NAME": ""
        }
    })
    wandb.run.config.update({
        'num_params': num_params,
    })

    # Synchronize all processes so that the directory is created before continuing
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if config.training.eval_only:
        # Start with evaluation
        if using_dist:
            torch.distributed.barrier()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_eval: nn.Module = model.module
        else:
            model_eval: nn.Module = model
        # save evaluation checkpoints and log metrics
        save_eval_callback_wandb(
            model_eval, optimizer, vals, 
            stats["epoch_in_training"], 
            config, rank, stats, 
            config.training.num_model_updates * config.training.num_agg * (stats["epoch_in_training"])
        )
        if using_dist:
            torch.distributed.barrier()
        if using_dist:
            cleanup()
        wandb.finish()
        return

    # Initialize the outer training loop
    for epoch in range(stats["epoch_in_training"] + 1, config.training.num_epochs + 1):
        LOGGER.text(f"Epoch {epoch}")

        # initialize the loss accumulators
        epoch_loss_cls = 0.0
        epoch_loss_reg = 0.0

        # set the model and optimizer to training mode
        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()

        # initialize the inner training loop
        for batch in tqdm(range(config.training.num_model_updates * config.training.num_agg)):
            # randomly set the evaluation position, i.e. the context length
                # ---- 1. choose a common eval_pos ---------------------------------------
            if using_dist and dist.get_rank() == 0:
                eval_pos_t = torch.randint(
                    config.training.min_eval_pos,
                    config.training.max_eval_pos + 1,
                    (1,),
                    device=device,
                    dtype=torch.long,
                )
            elif using_dist:
                eval_pos_t = torch.empty(1, dtype=torch.long, device=device)
            else:
                # Single GPU/CPU case - directly set eval_pos
                eval_pos = random.randint(config.training.min_eval_pos, config.training.max_eval_pos)

            if using_dist:
                dist.broadcast(eval_pos_t, src=0)
                eval_pos = int(eval_pos_t.item())

            # get the next batch from the DataLoader
            x, y, task = [a.to(device) for a in next(iter_data_loader)]

            # efficient forward pass and loss computation
            with selected_autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output, log_act_norms = model(
                    x, y.squeeze(-1)[:eval_pos], return_log_act_norms=True
                )

                # compute the losses
                loss_cls, loss_reg = compute_losses(output, task, y.squeeze(-1)[eval_pos:], config)

                # reweight the loss for combined classification and regression tasks
                loss = get_combined_loss(loss_cls, loss_reg, task, config)

            # detach the log_act_norms to avoid memory leak
            log_act_norms = {k: v.detach().item() for k, v in log_act_norms.items()}

            epoch_loss_cls += loss_cls.cpu().detach().item()
            epoch_loss_reg += loss_reg.cpu().detach().item()

            # backpropagation
            loss.backward()

            # gradient accumulation
            if (batch + 1) % config.training.num_agg == 0:
                # compute the global step
                global_step = (
                    batch + config.training.num_model_updates * config.training.num_agg * epoch
                )

                # log the losses and norms
                log_param_norms_wandb(model, global_step, task, global_step)

                # clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad_norm)

                # step the optimizer
                optimizer.step()
                optimizer.zero_grad()

        # log the average losses for the epoch
        wandb.log({
            "loss/cls-average_train": epoch_loss_cls / config.training.num_model_updates,
            "loss/reg-average_train": epoch_loss_reg / config.training.num_model_updates
        }, step=global_step)

        # evaluate the model every eval_every epochs
        if epoch % config.logging.eval_every == 0:
            # synchronize all processes before evaluation
            if using_dist:
                torch.distributed.barrier()

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_eval: nn.Module = model.module
            else:
                model_eval: nn.Module = model

            # save evaluation checkpoints and log metrics
            save_eval_callback_wandb(model_eval, optimizer, vals, epoch, config, rank, stats, global_step)

            if using_dist:
                torch.distributed.barrier()

    # cleanup the model and optimizer
    if using_dist:
        cleanup()

    wandb.finish()

if __name__ == "__main__":
    main()
