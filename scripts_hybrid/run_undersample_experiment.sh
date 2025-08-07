#!/bin/bash
# This script runs a series of supervised tasks using specified models and datasets.
# Usage: ./run_undesample_experiment.sh <task_name> <model_name> <data_dir> <embedders_dir> <embeddings_output_dir (root)>  <output_dir (root)>

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <task_name> <model_name> <data_dir> <embedders_dir> <embeddings_output_dir (root)> <output_dir (root)>"
    exit 1
fi

for n_samples in 10000 100000 250000; 
do
    echo "Running task: $1 with model: $2 using $n_samples samples"
    python3 scripts_hybrid/undersample_experiment.py \
        tasks@task=$1 \
        embedder=$2 \
        data_dir=$3 \
        embedders_dir=$4 \
        embeddings_output_dir=$5/$1/$2 \
        output_dir=$6/$1/$2/$n_samples \
        task.dataset.annotations_undersample=$n_samples
done