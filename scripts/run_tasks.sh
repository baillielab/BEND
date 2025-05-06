# This script runs the training for all tasks using the specified embedder.
# Usage: ./scripts/run_tasks.sh <embedder>

for task in gene_finding \
            enhancer_annotation \
            histone_modification \
            chromatin_accessibility \
            cpg_methylation
do
    python scripts/train_on_task.py --config-name $task embedder=$1
done
