#!/bin/bash
for model in nt_transformer_1000g \
            dnabert2 \
            nt_transformer_human_ref \
            nt_transformer_v2_500m \
            hyenadna-tiny-1k \
            hyenadna-large-1m \
            resnetlm \
            awdlstm
do
    for task in enhancer_annotation \
                gene_finding \
                histone_modification
    do
        echo "Running task: $task with model: $model"
        python3 scripts_batch/estimate_resources.py \
            data_dir=$1 \
            embedders_dir=$2 \
            embeddings_output_dir=$3/$task/$model/ \
            output_dir=$4 \
            tasks@task=$task \
            embedder=$model
    done
done
