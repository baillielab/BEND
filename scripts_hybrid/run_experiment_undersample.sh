#!/bin/bash
# This script runs a series of supervised tasks using specified models and datasets.
# Usage: ./run_undesample_experiment.sh <task_name> <model_name> <data_dir> <embedders_dir> <embeddings_output_dir (root)>  <output_dir (root)>


n_samples=500_000
echo "Running task: $1 with model: $2 using $n_samples samples"
python3 scripts_hybrid/experiment_undersample.py \
    tasks@task=$1 \
    embedder=$2 \
    data_dir=$3 \
    embedders_dir=$4 \
    embeddings_output_dir=$5/$1/$2 \
    output_dir=$6/$1/$2/$n_samples \
    task.dataset.num_train_annotations=$n_samples \
    task.data.num_train_embeddings=$n_samples \
    compute_embeddings=true
    
for n_samples in 10_000 100_000 250_000; 
do
    echo "Running task: $1 with model: $2 using $n_samples samples"
    python3 scripts_hybrid/experiment_undersample.py \
        tasks@task=$1 \
        embedder=$2 \
        data_dir=$3 \
        embedders_dir=$4 \
        embeddings_output_dir=$5/$1/$2 \
        output_dir=$6/$1/$2/$n_samples \
        task.data.num_train_embeddings=$n_samples \
        compute_embeddings=false
    
done