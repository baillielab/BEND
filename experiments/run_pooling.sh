#!/bin/bash
# This script runs a series of supervised tasks using specified models and datasets.
# Usage: . <script_path> <task_name> <model_name> <data_dir> <embedders_dir> <embeddings_output_dir (root)>  <output_dir (root)> <compute_embeddings> <run_without_pooling> <num_workers (default=32)>

task=$1
embedder=$2
data_dir=$3
embedders_dir=$4
embeddings_output_dir=$5
output_dir=$6
compute_embeddings=$7
run_without_pooling=$8
num_workers=${9:-32}


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

for mode in mean max mean_no_upsample swe;
do
    echo "Running task: $task with model: $embedder using $mode pooling"
    python3 scripts/train_downstream.py \
        task=$task \
        embedder=$embedder \
        data_dir=$data_dir \
        embedders_dir=$embedders_dir \
        embeddings_output_dir=$embeddings_output_dir \
        output_dir=$output_dir \
        task.model.pooling=$mode \
        task.dataloaders.num_workers=$num_workers
done

if [ $run_without_pooling = "true" ]; then
    echo "Running task: $task with model: $embedder without pooling"
    python3 scripts/train_downstream.py \
        task=$task \
        embedder=$embedder \
        data_dir=$data_dir \
        embedders_dir=$embedders_dir \
        embeddings_output_dir=$embeddings_output_dir \
        output_dir=$output_dir \
        task.model.pooling=default \
        task.dataloaders.num_workers=$num_workers
fi