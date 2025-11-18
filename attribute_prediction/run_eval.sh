#!/bin/bash

# run_eval.sh

# AGE datasets
echo "Run evaluation for age: given_age_everything_8b"
python age/evaluate_readmission.py \
  --embeddings-train-path age/artifacts/given_age_everything_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path age/artifacts/given_age_everything_8b/test/qwen3_embeddings.feather \
  --labels-train-path age/artifacts/given_age_everything_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path age/artifacts/given_age_everything_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path age/outputs/given_age_everything_8b.json

echo "Run evaluation for age: given_age_nothing_8b"
python age/evaluate_readmission.py \
  --embeddings-train-path age/artifacts/given_age_nothing_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path age/artifacts/given_age_nothing_8b/test/qwen3_embeddings.feather \
  --labels-train-path age/artifacts/given_age_nothing_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path age/artifacts/given_age_nothing_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path age/outputs/given_age_nothing_8b.json

echo "Run evaluation for age: no_age_everything_8b"
python age/evaluate_readmission.py \
  --embeddings-train-path age/artifacts/no_age_everything_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path age/artifacts/no_age_everything_8b/test/qwen3_embeddings.feather \
  --labels-train-path age/artifacts/no_age_everything_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path age/artifacts/no_age_everything_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path age/outputs/no_age_everything_8b.json

echo "Run evaluation for age: no_age_nothing_8b"
python age/evaluate_readmission.py \
  --embeddings-train-path age/artifacts/no_age_nothing_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path age/artifacts/no_age_nothing_8b/test/qwen3_embeddings.feather \
  --labels-train-path age/artifacts/no_age_nothing_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path age/artifacts/no_age_nothing_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path age/outputs/no_age_nothing_8b.json


# GENDER datasets
echo "Run evaluation for gender: given_gender_everything_8b"
python gender/evaluate_readmission.py \
  --embeddings-train-path gender/artifacts/given_gender_everything_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path gender/artifacts/given_gender_everything_8b/test/qwen3_embeddings.feather \
  --labels-train-path gender/artifacts/given_gender_everything_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path gender/artifacts/given_gender_everything_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path gender/outputs/given_gender_everything_8b.json

echo "Run evaluation for gender: given_gender_nothing_8b"
python gender/evaluate_readmission.py \
  --embeddings-train-path gender/artifacts/given_gender_nothing_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path gender/artifacts/given_gender_nothing_8b/test/qwen3_embeddings.feather \
  --labels-train-path gender/artifacts/given_gender_nothing_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path gender/artifacts/given_gender_nothing_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path gender/outputs/given_gender_nothing_8b.json

echo "Run evaluation for gender: no_gender_everything_8b"
python gender/evaluate_readmission.py \
  --embeddings-train-path gender/artifacts/no_gender_everything_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path gender/artifacts/no_gender_everything_8b/test/qwen3_embeddings.feather \
  --labels-train-path gender/artifacts/no_gender_everything_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path gender/artifacts/no_gender_everything_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path gender/outputs/no_gender_everything_8b.json

echo "Run evaluation for gender: no_gender_nothing_8b"
python gender/evaluate_readmission.py \
  --embeddings-train-path gender/artifacts/no_gender_nothing_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path gender/artifacts/no_gender_nothing_8b/test/qwen3_embeddings.feather \
  --labels-train-path gender/artifacts/no_gender_nothing_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path gender/artifacts/no_gender_nothing_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path gender/outputs/no_gender_nothing_8b.json


# RACE datasets
echo "Run evaluation for race: given_race_everything_8b"
python race/evaluate_readmission.py \
  --embeddings-train-path race/artifacts/given_race_everything_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path race/artifacts/given_race_everything_8b/test/qwen3_embeddings.feather \
  --labels-train-path race/artifacts/given_race_everything_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path race/artifacts/given_race_everything_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path race/outputs/given_race_everything_8b.json

echo "Run evaluation for race: given_race_nothing_8b"
python race/evaluate_readmission.py \
  --embeddings-train-path race/artifacts/given_race_nothing_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path race/artifacts/given_race_nothing_8b/test/qwen3_embeddings.feather \
  --labels-train-path race/artifacts/given_race_nothing_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path race/artifacts/given_race_nothing_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path race/outputs/given_race_nothing_8b.json

echo "Run evaluation for race: no_race_everything_8b"
python race/evaluate_readmission.py \
  --embeddings-train-path race/artifacts/no_race_everything_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path race/artifacts/no_race_everything_8b/test/qwen3_embeddings.feather \
  --labels-train-path race/artifacts/no_race_everything_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path race/artifacts/no_race_everything_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path race/outputs/no_race_everything_8b.json

echo "Run evaluation for race: no_race_nothing_8b"
python race/evaluate_readmission.py \
  --embeddings-train-path race/artifacts/no_race_nothing_8b/train/qwen3_embeddings.feather \
  --embeddings-test-path race/artifacts/no_race_nothing_8b/test/qwen3_embeddings.feather \
  --labels-train-path race/artifacts/no_race_nothing_8b/train/qwen3_embeddings_labels.feather \
  --labels-test-path race/artifacts/no_race_nothing_8b/test/qwen3_embeddings_labels.feather \
  --max-iter 1000 \
  --cv-folds 5 \
  --output-path race/outputs/no_race_nothing_8b.json

