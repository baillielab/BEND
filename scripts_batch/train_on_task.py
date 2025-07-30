"""
train_on_task.py
----------------
Train a model on a downstream task.
"""

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from bend.utils.task_trainer import (
    BaseTrainer,
    MSELoss,
    BCEWithLogitsLoss,
    PoissonLoss,
    CrossEntropyLoss,
)
import wandb
from bend.models.downstream import CustomDataParallel
import os
import sys
from bend_batch.utils import get_device, set_seed


set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


# load config
@hydra.main(config_path=f"../config", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a supervised task experiment.
    This function is called by hydra.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    # print(cfg.embedding.embedders)
    for key, value in cfg.items():
        print(f"{key}: {value}")

    device = get_device()

    os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
    print("output_dir", cfg.output_dir)

    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml")

    model = hydra.utils.instantiate(cfg.task.model).to(device).float()

    optimizer = hydra.utils.instantiate(cfg.task.optimizer, params=model.parameters())

    # define criterion
    print(f"Use {cfg.task.params.criterion} loss function")
    match cfg.task.params.criterion:
        case "cross_entropy":
            criterion = CrossEntropyLoss(
                ignore_index=cfg.task.data.padding_value,
                weight=(
                    torch.tensor(cfg.task.params.class_weights).to(device)
                    if cfg.task.params.class_weights is not None
                    else None
                ),
            )
        case "poisson_nll":
            criterion = PoissonLoss()
        case "mse":
            criterion = MSELoss()
        case "bce":
            criterion = BCEWithLogitsLoss(
                class_weights=(
                    torch.tensor(cfg.task.params.class_weights).to(device)
                    if cfg.task.params.class_weights is not None
                    else None
                )
            )

    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.task.data)

    # instantiate trainer
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg.task,
        overwrite_dir=True,
    )

    if cfg.task.params.mode == "train":
        trainer.train(
            train_loader,
            val_loader,
            test_loader,
            cfg.task.params.epochs,
            cfg.task.params.load_checkpoint,
        )

    trainer.test(test_loader, overwrite=False)


if __name__ == "__main__":
    print("Run experiment")
    run_experiment()
