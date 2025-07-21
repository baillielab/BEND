# !/bin/bash
# This script is used to run the training on the enhancer annotation task with a given embedder.

for fold in 0 1 2 3 4 5 6 7 8 9 
do
    python scripts/train_on_task.py --config-name=enhancer_annotation embedder=$1 data.cross_validation=$fold shuffle=$2
done