import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import bend.io.sequtils as sequtils
import pandas as pd
import numpy as np
import sys
from bend.utils.set_seed import set_seed, SEED
import time
import shutil

set_seed()
CSV_FILE_NAME = "embeddings_stats.csv"


def record_embedding_time(cfg, start_time: float, n_samples: int, size: float) -> None:
    """
    Record the time taken for embedding in a CSV file.
    Parameters
    ----------
    start_time : float
        The start time of the embedding process.
    """

    end_time = time.time()
    print(f"Embedding completed in {end_time - start_time:.2f} seconds")

    file_path = os.path.join(cfg.output_dir, CSV_FILE_NAME)

    if not os.path.exists(file_path):
        os.makedirs(cfg.output_dir, exist_ok=True)
        pd.DataFrame(
            columns=["task", "model", "time", "n_samples", "size (bytes)"],
        ).to_csv(file_path, index=False)

    data = pd.read_csv(file_path)
    data = data._append(
        {
            "task": cfg.task,
            "model": cfg.model,
            "time": end_time - start_time,
            "n_samples": n_samples,
            "size (bytes)": size,
        },
        ignore_index=True,
    )
    data.to_csv(file_path, index=False)


# load config
@hydra.main(
    config_path="../config_estimate/embedding/", config_name="embed", version_base=None
)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a embedding of nucleotide sequences.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    print("Embedding data for", cfg.task)

    split = "train" if cfg.task != "enhancer_annotation" else "part5"
    chunk = 0

    print(f"Overriding split to: {split} and chunk to: {chunk}")

    print("Embedding with", cfg.model)
    # instatiante model
    embedder = hydra.utils.instantiate(cfg[cfg.model])

    print(f"Embedding split: {split}")

    output_dir = f"{cfg.output_dir}/{cfg.task}/{cfg.model}/"
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # get length of bed file and divide by chunk size, if a spcific chunk is not set
    df = pd.read_csv(cfg[cfg.task].bed, sep="\t", low_memory=False)
    df = df[df.iloc[:, -1] == split] if split is not None else df

    n_samples = min(len(df), cfg.chunk_size) if cfg.chunk_size is not None else len(df)

    output_path = f"{output_dir}/{split}_{chunk}.tar.gz"

    print(f"\t Embedding chunk {chunk}")
    sequtils.embed_from_bed(
        **cfg[cfg.task],
        embedder=embedder,
        output_path=output_path,
        split=split,
        chunk=chunk,
        chunk_size=cfg.chunk_size,
        upsample_embeddings=(
            cfg[cfg.model]["upsample_embeddings"]
            if "upsample_embeddings" in cfg[cfg.model]
            else False
        ),
    )

    record_embedding_time(cfg, start_time, n_samples, os.path.getsize(output_path))


if __name__ == "__main__":

    print("Run Embedding")

    run_experiment()
