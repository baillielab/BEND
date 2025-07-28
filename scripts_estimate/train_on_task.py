"""
train_on_task.py
----------------
Train a model on a downstream task.
"""

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from bend.utils.task_trainer import (
    MSELoss,
    BCEWithLogitsLoss,
    PoissonLoss,
    CrossEntropyLoss,
)
from bend.estimate.task_trainer import EstimateTrainer
import wandb
from bend.models.downstream import CustomDataParallel
import os
import sys
from bend.utils.set_seed import set_seed
import time
import pandas as pd
import shutil

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


CSV_FILE_NAME = "dowstream_model_stats.csv"


# load config
@hydra.main(
    config_path=f"../config_estimate/supervised_tasks/",
    config_name=None,
    version_base=None,
)  #
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a supervised task experiment.
    This function is called by hydra.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    epochs = 1
    print(f"Override epochs to {epochs}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    print("output_dir", cfg.output_dir)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # instantiate model
    # initialization for supervised models
    if cfg.embedder == "resnet-supervised":
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.model.update(cfg.supervised_encoder[cfg.embedder])
    if cfg.embedder == "basset-supervised":
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.model.update(cfg.supervised_encoder[cfg.embedder])
    model = hydra.utils.instantiate(cfg.model).to(device).float()
    # put model on dataparallel
    if torch.cuda.device_count() > 1:
        from bend.models.downstream import CustomDataParallel

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = CustomDataParallel(model)
    # print(model)

    # print(torch.cuda.current_device())

    # instantiate optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # define criterion
    print(f"Use {cfg.params.criterion} loss function")
    if cfg.params.criterion == "cross_entropy":
        criterion = CrossEntropyLoss(
            ignore_index=cfg.data.padding_value,
            weight=(
                torch.tensor(cfg.params.class_weights).to(device)
                if cfg.params.class_weights is not None
                else None
            ),
        )
    elif cfg.params.criterion == "poisson_nll":
        criterion = PoissonLoss()
    elif cfg.params.criterion == "mse":
        criterion = MSELoss()
    elif cfg.params.criterion == "bce":
        criterion = BCEWithLogitsLoss(
            class_weights=(
                torch.tensor(cfg.params.class_weights).to(device)
                if cfg.params.class_weights is not None
                else None
            )
        )

    # init dataloaders
    if "supervised" in cfg.embedder:
        cfg.data.data_dir = cfg.data.data_dir.replace(cfg.embedder, "onehot")
    train_loader = hydra.utils.instantiate(cfg.data)  # instantiate dataloaders
    # instantiate trainer
    trainer = EstimateTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg,
        overwrite_dir=True,
    )

    if cfg.params.mode == "train":
        # train
        trainer.train(
            train_loader,
            None,
            None,
            epochs,
            False,  # do not load checkpoints
        )

    shutil.rmtree(cfg.data.data_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Run experiment")
    run_experiment()
