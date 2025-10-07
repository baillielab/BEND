#!/bin/bash
# This script runs a series of supervised tasks using specified models and datasets.
# Usage: . <script_path> <task_name> <model_name> <data_dir> <embedders_dir> <embeddings_output_dir (root)>  <output_dir (root)> <compute_embeddings> <num_workers (default=32)>

task=$1
embedder=$2
data_dir=$3
embedders_dir=$4
embeddings_output_dir=$5
output_dir=$6
compute_embeddings=$7
num_workers=${8:-32}


if [ $compute_embeddings = "true" ]; then
    echo "Embedding task: $task with model: $embedder"
    python3 scripts/compute_embeddings.py \
        task=$task \
        embedder=$embedder \
        data_dir=$data_dir \
        embedders_dir=$embedders_dir \
        embeddings_output_dir=$embeddings_output_dir \
        output_dir=$output_dir \
        task.dataloaders.num_workers=$num_workers
fi

for mode in none mean max min-max mean_no_upsample;
do
    echo "Running task: $task with model: $embedder using $mode pooling"
    python3 scripts/train_downstream.py \
        task=$task \
        embedder=$embedder \
        data_dir=$data_dir \
        embedders_dir=$embedders_dir \
        embeddings_output_dir=$embeddings_output_dir \
        output_dir=$output_dir \
        pooling_mode=$mode \
        task.dataloaders.num_workers=$num_workers
done