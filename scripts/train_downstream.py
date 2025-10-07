"""
Run a supervised task experiment by embedding DNA sequences, using compute_embeddings(),
and training a downstream model, using train_downstream().
Configuration is set through Hydra in config/config.yaml.
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from bend_hybrid.downstream.dataloaders import (
    get_samples_idx_by_split,
    undersample_dataloaders,
)
from bend_hybrid.downstream.trainer import BaseTrainer
from bend_hybrid.embedding.pooling import PoolingMode
from bend_hybrid.utils import get_device, set_seed, wandb_login

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def set_pooling_mode(cfg: DictConfig) -> str:
    """
    Set and validate the pooling mode in the configuration.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    Returns
    -------
    str
        Validated pooling mode.
    Raises
    ------
    ValueError
        If the pooling mode is unknown or incompatible with the embedder.
    """

    pooling_mode = (
        PoolingMode.NONE.value if cfg.pooling_mode is None else cfg.pooling_mode
    )

    if pooling_mode not in [mode.value for mode in PoolingMode]:
        raise ValueError(
            f"Unknown pooling mode: {cfg.pooling_mode}. Valid modes are {[mode.value for mode in PoolingMode]}"
        )

    if (
        pooling_mode == PoolingMode.MEAN_NO_UPSAMPLE.value
        and not cfg.embedding[cfg.embedder].upsample_embeddings
    ):
        raise ValueError(
            "Pooling mode 'mean_no_upsample' is not compatible with an embedder that does not upsample embeddings."
        )

    cfg.pooling_mode = pooling_mode


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

    # Gene finding and enhancer tasks downsample in the downstream model and do not support pooling.
    # Other tasks will downsample in the downstream model by their sequence length, if pooling mode is NONE.
    output_downsample_window = cfg.task.model.get("output_downsample_window", None)
    if cfg.pooling_mode == PoolingMode.NONE.value and output_downsample_window is None:
        output_downsample_window = cfg.task.dataset.get("sequence_length", None)

    model = (
        hydra.utils.instantiate(
            cfg.task.model, output_downsample_window=output_downsample_window
        )
        .to(device)
        .float()
    )
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

    set_pooling_mode(cfg)
    print("Using pooling mode:", cfg.pooling_mode)

    wandb_login()
    wandb.init(
        anonymous="allow",
        project="bend-pooling",
        name=f"{cfg.task.name}_{cfg.embedder}_{cfg.pooling_mode}",
        config=OmegaConf.to_container(cfg),
    )
    wandb.summary["task"] = cfg.task.name
    wandb.summary["embedder"] = cfg.embedder
    wandb.summary["downstream/pooling_mode"] = cfg.pooling_mode

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task.name, cfg.embedder, cfg.pooling_mode
    )
    cfg.output_dir = os.path.join(
        cfg.output_dir,
        cfg.task.name,
        cfg.embedder,
        cfg.pooling_mode,
        "downstream",
    )
    os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
    print("output_dir", cfg.output_dir)

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
