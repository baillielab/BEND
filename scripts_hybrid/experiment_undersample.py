import glob
import math
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
from bend_hybrid.data_downstream import return_dataloader
from bend_hybrid.datasets import DEFAULT_SPLIT_COLUMN_IDX, DataSupervised, collate_fn
from bend_hybrid.utils import get_device, record_embedding_time, set_seed

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def embed(cfg: DictConfig, split: str, frac_annotations: float = None) -> None:
    """
    Embed all sequences in the dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)
    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    print("Loading dataset ...")
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
        frac=frac_annotations,
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


def train_on_task(cfg: DictConfig, n_samples_train=None, n_samples_valid=None) -> None:
    """
    Train the downstream model of a supervised task.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    device = get_device()

    os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
    print("output_dir", cfg.output_dir)

    model = hydra.utils.instantiate(cfg.task.model).to(device).float()

    optimizer = hydra.utils.instantiate(cfg.task.optimizer, params=model.parameters())

    tars = glob.glob(f"{cfg.task.data.data_dir}/*.tar.gz")
    dataloaders = []
    for split, n_samples in zip(
        ["train", "valid", "test"], [n_samples_train, n_samples_valid, None]
    ):

        data = [x for x in tars if os.path.split(x)[-1].startswith(split)]
        dataloader = return_dataloader(
            data=data,
            batch_size=cfg.task.data.batch_size,
            num_workers=cfg.task.data.num_workers,
            padding_value=cfg.task.data.padding_value,
            shuffle=cfg.task.data.shuffle,
            shardshuffle=False,  # Do not shuffle shards, but train on the same data
        )

        if n_samples is not None:
            dataloader.with_epoch(math.ceil(n_samples / cfg.task.data.batch_size))

        dataloaders.append(dataloader)

    if len(dataloaders) != 3:
        raise ValueError(
            "Expected three dataloaders for train, valid, and test splits, but got "
            f"{len(dataloaders)}."
        )

    train_loader, val_loader, test_loader = dataloaders

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

    print("Retrieving splits from annotations...")
    annotations = pd.read_csv(
        cfg.task.dataset.annotations_path, sep="\t", low_memory=False
    )
    splits = annotations.iloc[:, DEFAULT_SPLIT_COLUMN_IDX].unique().tolist()
    train_annotations = annotations[
        annotations.iloc[:, DEFAULT_SPLIT_COLUMN_IDX] == "train"
    ]
    if len(train_annotations) == 0:
        raise ValueError(
            "No training annotations found. Cannot undersample train and validation splits."
        )

    if cfg.compute_embeddings is True:
        print(
            f"=== Embedding sequences for task: {cfg.task.task} with model: {cfg.embedder} ==="
        )

        frac_annotations = None
        # Convert num_train_annotations to fraction
        if "num_train_annotations" in cfg.task.dataset:
            frac_annotations = min(
                1.0,
                cfg.task.dataset.num_train_annotations / len(train_annotations),
            )

            if frac_annotations >= 1.0:
                print(
                    f"Warning: num_train_annotations ({cfg.task.dataset.num_train_annotations}) "
                    "is greater or equal to the total number of training annotations. "
                    "Using all training annotations."
                )
                frac_annotations = None

        start_time = time.time()
        for split in splits:
            print(f"=== Processing split: {split} ===")
            embed(cfg, split, frac_annotations if split != "test" else None)

        record_embedding_time(
            cfg.task.task,
            cfg.embedder,
            running_time=time.time() - start_time,
            n_samples=len(annotations),
            tar_path=cfg.embeddings_output_dir,
            output_dir=cfg.output_dir,
        )

    n_samples_train = None
    n_samples_valid = None

    if (
        "num_train_embeddings" in cfg.task.data
        and cfg.task.data.num_train_embeddings is not None
    ):
        valid_annotations = annotations[
            annotations.iloc[:, DEFAULT_SPLIT_COLUMN_IDX] == "valid"
        ]

        n_samples_train = max(1, cfg.task.data.num_train_embeddings)
        if n_samples_train > len(train_annotations):
            print(
                f"Warning: num_train_embeddings ({n_samples_train}) is greater than the "
                "total number of training annotations. Using all training annotations."
            )
            n_samples_train = len(train_annotations)

        undersampling_ratio = n_samples_train / len(train_annotations)
        n_samples_valid = max(
            1, math.ceil(undersampling_ratio * len(valid_annotations))
        )

    train_on_task(cfg, n_samples_train, n_samples_valid)


if __name__ == "__main__":
    run_experiment()
