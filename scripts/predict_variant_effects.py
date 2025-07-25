"""
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
"""

import argparse
import time

import hydra
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
from scipy import spatial
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import yaml


def main():

    parser = argparse.ArgumentParser("Compute embeddings")
    parser.add_argument("--work_dir", type=str, help="Path to the data directory")
    parser.add_argument(
        "--type",
        choices=["expression", "disease"],
        type=str,
        help="Type of variant effects experiment (expression or disease)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model architecture for computing embeddings",
    )

    cfg = yaml.safe_load(
        open(
            os.path.join(
                parser.parse_args().work_dir, "conf", "embedding", "embed.yaml"
            )
        )
    )

    args = parser.parse_args()

    experiment_name = f"variant_effects_{args.type}"

    annotation_path = os.path.join(
        args.work_dir, "data", "variant_effects", f"{experiment_name}.bed"
    )
    reference_path = os.path.join(
        args.work_dir, "data", "genomes", "GRCh38.primary_assembly.genome.fa"
    )

    output_dir = os.path.join(args.work_dir, "results", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if "model_path" in cfg[args.model].keys():
        cfg[args.model]["model_path"] = cfg[args.model]["model_path"].replace(
            "${embedders_dir}", os.path.join(args.work_dir, cfg["embedders_dir"])
        )
    print("Loading embedder", args.model)

    embedder = hydra.utils.instantiate(cfg[args.model])
    kwargs = {"disable_tqdm": True}
    embedding_idx = 256
    extra_context = extra_context_left = extra_context_right = 256

    if "awdlstm" in args.model or "hyenadna" in args.model:
        embedding_idx = 511
        extra_context = 512
        # autogressive model. No use for right context.
        extra_context_left = extra_context
        extra_context_right = 0

    if "nt" in args.model or "dnabert2" in args.model:
        kwargs["upsample_embeddings"] = True  # each nucleotide has an embedding

    print("Loading genome data")
    genome_annotation = Annotation(annotation_path, reference_genome=reference_path)
    if extra_context > 0:
        genome_annotation.extend_segments(
            extra_context_left=extra_context_left,
            extra_context_right=extra_context_right,
        )
    genome_annotation.annotation["distance"] = 0.0

    start = time.time()

    # iterate over the genome annotation
    for index, row in tqdm(genome_annotation.annotation.iterrows()):

        # get the reference and alternate dna sequences
        dna = genome_annotation.get_dna_segment(index=index)
        dna_alt = [x for x in dna]
        if extra_context_left == extra_context_right:
            dna_alt[len(dna_alt) // 2] = row["alt"]
        elif extra_context_right == 0:
            dna_alt[-1] = row["alt"]
        elif extra_context_left == 0:
            dna_alt[0] = row["alt"]
        else:
            raise ValueError("Not implemented")
        dna_alt = "".join(dna_alt)

        # compute the embeddings
        embedding_wt, embedding_alt = embedder.embed([dna, dna_alt], **kwargs)
        embedding_wt = embedding_wt[0, embedding_idx]
        embedding_alt = embedding_alt[0, embedding_idx]

        # compute the cosine distance
        d = spatial.distance.cosine(embedding_alt, embedding_wt)
        genome_annotation.annotation.loc[index, "distance"] = d

    running_time = time.time() - start
    print(f"Finished computing embeddings in {running_time:.2f} seconds")

    roc_auc = roc_auc_score(
        genome_annotation.annotation["label"], genome_annotation.annotation["distance"]
    )
    print(f"ROC AUC: {roc_auc} for {args.model}")

    print(f"Saving cosine distances...")
    genome_annotation.annotation.to_csv(
        output_dir + f"/distances_{args.model}.csv", index=False
    )

    # save the results
    path_results_df = os.path.join(output_dir, "results_rocauc.csv")
    if os.path.exists(path_results_df):
        results_df = pd.read_csv(path_results_df)
    else:
        results_df = pd.DataFrame(
            {
                "model": [],
                "roc_auc": [],
                "running_time": [],
            }
        )
    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                {
                    "model": [args.model],
                    "roc_auc": [roc_auc],
                    "running_time": [running_time],
                }
            ),
        ],
        ignore_index=True,
    )

    results_df.to_csv(path_results_df, index=False)


if __name__ == "__main__":
    main()
