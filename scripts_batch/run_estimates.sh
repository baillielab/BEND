#!/bin/bash
for model in hyenadna-tiny-1k \
            hyenadna-large-1m \
            resnetlm \
            nt_transformer_human_ref \
            nt_transformer_1000g \
            dnabert2 \
            nt_transformer_v2_500m \
            awdlstm
do
    for task in gene_finding \
                enhancer_annotation \
                histone_modification
    do
        echo "Running task: $task with model: $model"
        python3 scripts_batch/estimate_resources.py \
            embeddings_output_dir=$1/$task/$model/ \
            output_dir=$1 \
            tasks@task=$task \
            embedder=$model
    done
done
