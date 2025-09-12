"""
Script for hyperparameter tuning of the variant effect prediction task.
To run, use the following command:
`python scripts/sweep_variant_effects.py --multirun +sweep=var_effects_expression`
"""

import os
import time

import hydra

from omegaconf import DictConfig

from torch.utils.data import DataLoader

from bend_hybrid.embedding.datasets import DataVariantEffects
from bend_hybrid.utils import set_seed


set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


@hydra.main(config_path="../config/", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run the experiment.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    embedding_idx = 256
    extra_context_left = extra_context_right = 256
    if embedder.autoregressive:
        embedding_idx = 511
        extra_context_left = 512
        extra_context_right = 0

    dataset = DataVariantEffects(
        annotation_path=cfg.task.dataset.annotations_path,
        genome_path=cfg.task.dataset.genome_path,
        extra_context_left=extra_context_left,
        extra_context_right=extra_context_right,
        n_samples=cfg.get("n_samples", None),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.task.dataloader.batch_size,
        num_workers=cfg.task.dataloader.num_workers,
        prefetch_factor=(
            cfg.task.dataloader.prefetch_factor
            if cfg.task.dataloader.prefetch_factor > 0
            else None
        ),
        shuffle=False,
    )

    start = time.time()

    for _, _ in enumerate(dataloader):
        pass

    end = time.time()
    return end - start


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
