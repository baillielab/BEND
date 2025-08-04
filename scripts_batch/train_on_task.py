"""
train_on_task.py
----------------
Train a model on a downstream task.
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from bend.utils.task_trainer import BaseTrainer
from bend_batch.utils import get_device, set_seed

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


# load config
@hydra.main(config_path="../config", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a supervised task experiment.
    This function is called by hydra.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    device = get_device()

    os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
    print("output_dir", cfg.output_dir)

    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml")

    model = hydra.utils.instantiate(cfg.task.model).to(device).float()

    optimizer = hydra.utils.instantiate(cfg.task.optimizer, params=model.parameters())

    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.task.data)

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
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
