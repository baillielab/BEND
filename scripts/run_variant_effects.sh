# This script runs the variant effect prediction experiments.
# For each model, we extract the cosine distance between the reference and mutated sequence embedding.

# Run expression variants.
WORK_DIR=${1:-./}

# baselines
python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type expression --model convnet
python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type disease --model convnet

python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type expression --model awdlstm
python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type disease --model awdlstm

# hyenadna
python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type expression --model hyenadna
python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type disease --model hyenadna

# nt
python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type expression --model nt
python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type disease --model nt

# # dnabert2
# python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type expression --model dnabert2
# python3 scripts/predict_variant_effects.py --work_dir $WORK_DIR --type disease --model dnabert2
