#!/bin/bash

for model in hyenadna-tiny-1k \
            hyenadna-large-1m \
            dnabert6 \
            resnetlm \
            nt_transformer_ms \
            nt_transformer_human_ref \
            nt_transformer_1000g \
            dnabert2 \
            nt_transformer_v2_500m \
            awdlstm
do
    for task in gene_finding \
                enhancer_annotation \
                histone_modification \
                chromatin_accessibility \
                cpg_methylation
    do
        echo "Running task: $task with model: $model"
        python3 scripts_estimate/precompute_embeddings.py \
            work_dir=$1 \
            output_dir=$2 \
            task=$task \
            model=$model
        python3 scripts_estimate/train_on_task.py \
            embedder=$model \
            data.data_dir=$2/$task/$model/ \
            output_dir=$2 \
            --config-name=$task
    done
done
