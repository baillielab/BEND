"""
This script differs from the default run_supervised_script.py script in that it
computes embeddings for two sequences: the reference sequence and the variant sequence.
The variant sequence is obtained by replacing the reference nucleotide with the variant
nucleotide at the variant position.
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf
from scipy import spatial
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from bend_batch.embedding.datasets import DataVariantEffects
from bend_batch.utils import set_seed, wandb_login, log_time

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

    cfg.output_dir = os.path.join(cfg.output_dir, cfg.task.name, cfg.embedder)
    print("Output directory", cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    wandb_login()
    wandb.init(
        anonymous="allow",
        project=cfg.wandb.project,
        name=f"{cfg.task.name}_{cfg.embedder}",
        config=OmegaConf.to_container(cfg),
    )
    wandb.summary["task"] = cfg.task.name
    wandb.summary["embedder"] = cfg.embedder

    print(f"Computing embeddings for {cfg.task.name} using {cfg.embedder}")
    embedder = hydra.utils.instantiate(
        cfg.embedding[cfg.embedder], task_input_length=cfg.task.dataset.sequence_length
    )

    embedding_idx = 256
    extra_context_left = extra_context_right = 256
    if embedder.autoregressive:
        print("Using autoregressive embedding")
        embedding_idx = 511
        extra_context_left = 512
        extra_context_right = 0

    print("Loading genome data")
    dataset = DataVariantEffects(
        annotation_path=cfg.task.dataset.annotations_path,
        genome_path=cfg.task.dataset.genome_path,
        extra_context_left=extra_context_left,
        extra_context_right=extra_context_right,
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

    with log_time("running_time", log_type="summary"):
        cosine_distances = []
        labels = []

        for _, (dna_seqs, alt_dna_seqs, batch_labels) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            ref_embeddings = embedder(dna_seqs)[:, embedding_idx, :]
            snp_embeddings = embedder(alt_dna_seqs)[:, embedding_idx, :]

            for ref_emb, snp_emb, label in zip(
                ref_embeddings, snp_embeddings, batch_labels
            ):
                cosine_distances.append(spatial.distance.cosine(ref_emb, snp_emb))
                labels.append(label)

    score = roc_auc_score(labels, cosine_distances)
    print(f"ROC AUC: {score} for {cfg.embedder}")
    wandb.summary["downstream/test/metric"] = score

    wandb.finish()


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
