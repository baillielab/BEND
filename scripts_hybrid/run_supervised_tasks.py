import os
import time

import hydra
import numpy as np
import pandas as pd
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bend.utils.task_trainer import BaseTrainer
from bend_hybrid.datasets import DEFAULT_SPLIT_COLUMN_IDX, DataSupervised, collate_fn
from bend_hybrid.embedders.embedders import BaseEmbedder
from bend_hybrid.utils import get_device, record_embedding_time, set_seed

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def embed(cfg: DictConfig, embedder: BaseEmbedder, split: str) -> None:
    """
    Embed all sequences in the dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    embedder : BaseEmbedder
        The embedder to use for embedding sequences.
    split : str
        The dataset split to embed (e.g., "train", "val", "test").
    """

    dataset = DataSupervised(
        annotations_path=cfg.task.dataset.annotations_path,
        genome_path=cfg.task.dataset.genome_path,
        label_depth=(
            cfg.task.dataset.label_depth if "label_depth" in cfg.task.dataset else None
        ),
        hdf5_path=(
            cfg.task.dataset.hdf5_path if "hdf5_path" in cfg.task.dataset else None
        ),
        sequence_length=cfg.task.dataset.sequence_length,
        split=split,
    )

    is_data_uneven = True if cfg.task.dataset.sequence_length is None else False

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.task.data.batch_size,
        num_workers=cfg.task.data.num_workers,
        shuffle=True if split == "train" else False,
        collate_fn=collate_fn if is_data_uneven else None,
    )

    with wds.ShardWriter(
        os.path.join(cfg.embeddings_output_dir, f"{split}_%06d.tar.gz"),
        verbose=0,
        compress="gz",
    ) as writer:
        for batch_idx, (sequences, labels) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Embedding {split}"
        ):
            embeddings = embedder(sequences, uneven_length=is_data_uneven)

            for sample_idx in tqdm(
                range(len(embeddings)), desc="Writing samples", leave=False
            ):
                sample_key = batch_idx * cfg.task.data.batch_size + sample_idx
                writer.write(
                    {
                        "__key__": f"sample{sample_key:08d}",
                        "input.npy": embeddings[sample_idx],
                        "output.npy": np.array(labels[sample_idx], dtype=np.int32),
                    }
                )


def train_on_task(cfg: DictConfig) -> None:
    """
    Train the downstream model of a supervised task.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    device = get_device()

    cfg.output_dir = os.path.join(cfg.output_dir, "downstream")
    os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
    print("output_dir", cfg.output_dir)

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

    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml")

    if cfg.task.params.mode == "train":
        trainer.train(
            train_loader,
            val_loader,
            test_loader,
            cfg.task.params.epochs,
            cfg.task.params.load_checkpoint,
        )

    trainer.test(test_loader, overwrite=False)


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

    if "var" in cfg.task.task:
        print(
            "Skipping experiment for task variant_effects, as it is not a supervised task."
        )
        return

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task.task, cfg.embedder
    )
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.task.task, cfg.embedder)

    print("Retrieving splits from annotations...")
    annotations = pd.read_csv(
        cfg.task.dataset.annotations_path, sep="\t", low_memory=False
    )
    splits = annotations.iloc[:, DEFAULT_SPLIT_COLUMN_IDX].unique().tolist()

    if cfg.compute_embeddings is True:
        print(
            f"=== Embedding sequences for task: {cfg.task.task} with model: {cfg.embedder} ==="
        )

        os.makedirs(cfg.embeddings_output_dir, exist_ok=True)
        embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

        start_time = time.time()

        for split in splits:
            print(f"=== Processing split: {split} ===")
            embed(cfg, embedder, split)

        record_embedding_time(
            cfg.task.task,
            cfg.embedder,
            running_time=time.time() - start_time,
            n_samples=len(annotations),
            tar_path=cfg.embeddings_output_dir,
            output_dir=cfg.output_dir,
        )

    if cfg.train_model is True:
        print(
            f"=== Training model for task: {cfg.task.task} with embedder: {cfg.embedder} ==="
        )
        if (
            "cross_validation" in cfg.task.data.keys()
            and cfg.task.data.cross_validation is True
        ):
            output_dir = cfg.output_dir

            for fold in range(len(splits)):
                print(f"=== Running fold {fold + 1}/{len(splits)} ===")
                cfg.task.data.cross_validation = fold
                cfg.output_dir = os.path.join(output_dir, f"split_{fold + 1}")
                train_on_task(cfg)
        else:
            train_on_task(cfg)


if __name__ == "__main__":
    run_experiment()
