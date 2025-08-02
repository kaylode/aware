import random
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
import torch.autograd as autograd

from datasets import FullDataset, collate_fn
from model import TabDPTModel
from utils import (
    compute_losses,
    get_combined_loss,
    seed_everything,
)



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


    seed_everything(config.seed)

    checkpoint = torch.load(config.exp_path, weights_only=False)
    model_state = checkpoint["model"]

    model = TabDPTModel.load(model_state, config=checkpoint["cfg"])
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.{2}f} M parameters")

    return model


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(config: DictConfig):
    """Main function to run the training process.

    Args:
        config (DictConfig): The configuration for the training process.
    """

    # print config
    print("Config:", OmegaConf.to_yaml(config))

    # distributed training
    device = torch.device("cuda:0")

    model = set_experiment_from_config(config)
    model.to(device)


    # select the device type
    device_type = "cpu" if config.env.device == "cpu" else "cuda"

    # Use autocast for mixed precision training
    selected_autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    # Initialize the FullDataset for training
    print("Initializing Dataset...")
    dataset = FullDataset(device, config)

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

    eval_pos = random.randint(config.training.min_eval_pos, config.training.max_eval_pos)

    # get the next batch from the DataLoader
    x, y, task = [a.to(device) for a in next(iter_data_loader)]
    model.eval()
    grads = []

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

        breakpoint()

        grad_params = autograd.grad(loss, x, create_graph=True, allow_unused=True)
        grads.append(grad_params[0].detach().cpu())

    grads = torch.cat(grads)
    importances = grads.pow(2).sum(dim=0).pow(1/2.)

    print(f"Feature importances: {importances}")



if __name__ == "__main__":
    main()
