"""
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
"""

import hydra
from omegaconf import DictConfig
import os
from bend_batch.utils import set_seed, get_device
from bend_batch.datasets import DataVariantEffects
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import time
from scipy import spatial
from tqdm.auto import tqdm

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


@hydra.main(config_path=f"../config/", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:

    device = get_device()

    cfg_task = cfg.tasks[cfg.task]

    cfg.output_dir = os.path.join(cfg.output_dir, cfg.task, cfg.embedder)
    print("Output directory", cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Computing embeddings for {cfg.task} using {cfg.embedder}")
    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    embedding_idx = 256
    extra_context_left = extra_context_right = 256
    if embedder.autoregressive:
        print("Using autoregressive embedding")
        embedding_idx = 511
        extra_context_left = 512
        extra_context_right = 0

    print("Loading genome data")
    dataset = DataVariantEffects(
        annotation_path=cfg_task.dataset.annotations,
        genome_path=cfg_task.dataset.genome,
        extra_context_left=extra_context_left,
        extra_context_right=extra_context_right,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg_task.dataloader.batch_size,
        num_workers=cfg_task.dataloader.num_workers,
        shuffle=False,
    )

    start = time.time()
    cosine_dinstances = []

    for batch_idx, (dna_seqs, alt_dna_seqs) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        ref_embeddings = embedder.embed(dna_seqs)[:, embedding_idx, :]
        alt_embeddings = embedder.embed(alt_dna_seqs)[:, embedding_idx, :]

        for ref_emb, alt_emb in zip(ref_embeddings, alt_embeddings):
            cosine_dinstances.append(spatial.distance.cosine(ref_emb, alt_emb))

    dataset.annotation.loc[0 : len(cosine_dinstances) - 1, "distance"] = (
        cosine_dinstances
    )
    end = time.time()
    print(f"Running time: {end - start} seconds")

    score = roc_auc_score(dataset.annotation["label"], dataset.annotation["distance"])
    print(f"ROC AUC: {score} for {cfg.embedder}")

    # save the results
    pd.DataFrame(
        {"model": [cfg.embedder], "roc_auc": [score], "time": [end - start]}
    ).to_csv(os.path.join(cfg.output_dir, "roc_auc_scores.csv"), index=False)

    dataset.annotation.to_csv(
        os.path.join(cfg.output_dir, "distances.csv"), index=False
    )


if __name__ == "__main__":
    run_experiment()
