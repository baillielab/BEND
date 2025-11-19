"""
Run a supervised task experiment by embedding DNA sequences, using compute_embeddings(),
and training a downstream model, using train_downstream().
Configuration is set through Hydra in config/config.yaml.
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from bend_batch.downstream.dataloaders import (
    get_samples_idx_by_split,
    undersample_dataloaders,
)
from bend_batch.downstream.trainer import BaseTrainer
from bend_batch.utils import get_device, set_seed, wandb_login

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def train_cross_validation(cfg: DictConfig) -> None:
    fold_names = list(
        get_samples_idx_by_split(cfg.task.dataset.annotations_path).keys()
    )

    if cfg.task.dataloaders.fold_idx is None:
        # When fold_idx is None, perform cross-validation across all folds
        output_dir = cfg.output_dir
        for fold_idx, fold_test in enumerate(fold_names):
            print(f"=== Running fold {fold_idx + 1}/{len(fold_names)} ===")

            cfg.output_dir = os.path.join(output_dir, fold_test)

            val_idx = fold_idx + 1 if fold_idx + 1 < len(fold_names) else 0
            fold_valid = fold_names[val_idx]

            train_downstream(cfg, (fold_test, fold_valid))
    else:
        # When fold_idx is specified, use the specified fold for testing
        fold_idx = cfg.task.dataloaders.fold_idx
        if fold_idx < 0 or fold_idx >= len(fold_names):
            raise ValueError(f"fold_idx {fold_idx} is out of range.")
        fold_test = fold_names[fold_idx]
        val_idx = fold_idx + 1 if fold_idx + 1 < len(fold_names) else 0
        fold_valid = fold_names[val_idx]

        train_downstream(cfg, (fold_test, fold_valid))


def train_downstream(
    cfg: DictConfig,
    test_valid_folds: tuple[str, str] | None = None,
) -> None:
    """
    Train the downstream model of a supervised task.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    test_valid_folds : tuple[str, str], optional
        Tuple containing the fold names to use as test and validation sets. The default is None.
        Example: ('part0', 'part1') will use all tar files containing 'part0' as test set
        and all tar files containing 'part1' as validation set.
    """

    device = get_device()

    model = hydra.utils.instantiate(cfg.task.model).to(device).float()
    optimizer = hydra.utils.instantiate(cfg.task.optimizer, params=model.parameters())

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=cfg.task,
        overwrite_dir=True,
    )

    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml")

    dataloaders = hydra.utils.instantiate(
        cfg.task.dataloaders, test_valid_folds=test_valid_folds
    )

    n_samples = cfg.task.dataset.get("n_samples", None)
    if n_samples is not None:
        dataloaders = undersample_dataloaders(
            cfg,
            dataloaders,
            n_samples,
            test_valid_folds=test_valid_folds,
        )

    trainer.train(
        dataloaders["train"],
        dataloaders["valid"],
        cfg.task.params.epochs,
        cfg.task.params.load_checkpoint,
    )

    trainer.test(dataloaders["test"], overwrite=False)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run the experiment.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task.name, cfg.embedder
    )

    if not os.path.exists(cfg.embeddings_output_dir):
        raise ValueError(
            f"Embeddings directory {cfg.embeddings_output_dir} does not exist. "
            "If the selected pooling mode is supported by embedder, please compute embeddings first by running scripts/compute_embeddings.py"
        )

    cfg.output_dir = os.path.join(
        cfg.output_dir,
        cfg.task.name,
        cfg.embedder,
        "downstream",
    )
    os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
    print("output_dir", cfg.output_dir)

    wandb_login()
    wandb.init(
        anonymous="allow",
        project=cfg.wandb.project,
        name=f"{cfg.task.name}_{cfg.embedder}",
        config=OmegaConf.to_container(cfg),
    )
    wandb.summary["task"] = cfg.task.name
    wandb.summary["embedder"] = cfg.embedder

    print(
        f"=== Training model for task: {cfg.task.name} with embedder: {cfg.embedder} ==="
    )

    if "fold_idx" in cfg.task.dataloaders:
        train_cross_validation(cfg)
    else:
        train_downstream(cfg)
    wandb.finish()


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
