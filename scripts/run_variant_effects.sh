# This script runs the variant effect prediction experiments.
# For each model, we extract the cosine distance between the reference and mutated sequence embedding.

# Run expression variants.
WORK_DIR=${1:-./}

for type in expression disease
do
    echo "Running variant effects for type: $type"
    # baselines
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model resnetlm
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model awdlstm

    # hyenadna
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model hyenadna-tiny-1k
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model hyenadna-large-1m

    # nt
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt_transformer_1000g
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt_transformer_ms
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt_transformer_human_ref
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt_transformer_v2_500m

    # dnabert2
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model dnabert2

done

