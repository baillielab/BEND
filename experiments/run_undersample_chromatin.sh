#!/bin/bash
# This script runs a series of supervised tasks using specified models and datasets.
# Usage: . <script_path> <model_name> <data_dir> <embedders_dir> <embeddings_output_dir (root)>  <output_dir (root)> <compute_embeddings>

task=chromatin_accessibility
max_samples=700000
embedder=$1
data_dir=$2
embedders_dir=$3
embeddings_output_dir=$4 #/$task/$1
output_dir=$5 #/$task/$1
compute_embeddings=$6


if [ $compute_embeddings = "true" ]; then
    echo "Embedding task: $task with model: $embedder using $max_samples samples"
    python3 scripts/run_supervised_tasks.py \
        task=$task \
        embedder=$embedder \
        data_dir=$data_dir \
        embedders_dir=$embedders_dir \
        embeddings_output_dir=$embeddings_output_dir \
        output_dir=$output_dir \
        task.dataset.n_samples=$n_samples \
        compute_embeddings=true \
        train_downstream=false
fi

for n_samples in 10000 50000 100000 150000 250000 500000 700000;
do
    echo "Running task: $task with model: $embedder using $n_samples samples"
    python3 scripts/run_supervised_tasks.py \
        task=$task \
        embedder=$embedder \
        data_dir=$data_dir \
        embedders_dir=$embedders_dir \
        embeddings_output_dir=$embeddings_output_dir \
        output_dir=$output_dir/$n_samples \
        task.dataset.n_samples=$n_samples \
        task.dataloaders.shuffle=null \
        task.dataloaders.shardshuffle=false \
        compute_embeddings=false \
        train_downstream=true
done