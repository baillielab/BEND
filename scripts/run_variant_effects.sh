# This script runs the variant effect prediction experiments.
# For each model, we extract the cosine distance between the reference and mutated sequence embedding.

# Run expression variants.
WORK_DIR=${1:-./}

for type in expression disease
do
    echo "Running variant effects for type: $type"
    # baselines
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model resnetlm --version resnetlm
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model awdlstm --version awdlstm

    # hyenadna
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model hyenadna --version hyenadna-tiny-1k-seqlen
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model hyenadna --version hyenadna-large-1m-seqlen

    # nt
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt --version nucleotide-transformer-500m-1000g
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt --version nucleotide-transformer-2.5b-1000g
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt --version nucleotide-transformer-2.5b-multi-species
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt --version nucleotide-transformer-500m-human-ref
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model nt --version nucleotide-transformer-v2-500m-multi-species
    
    # dnabert2
    python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type $type --model dnabert2 --version dnabert2

done

