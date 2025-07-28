# !/bin/bash
# This script is used to run the training on the enhancer annotation task with a given embedder.

for fold in 0 1 2 3 4 5 6 7 8 9
do
    python3 scripts/train_on_task.py work_dir=$1 data.data_dir=$2 embedder=$3 data.cross_validation=$fold shuffle=$4 data.num_workers=$5 data.batch_size=$6 params.load_checkpoints=$7 --config-name=enhancer_annotation
done