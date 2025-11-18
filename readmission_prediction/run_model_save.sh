#!/bin/bash

# run_eval.sh

datasets=(
    "_name_address_age_visitdates_gender_race_"
    "_"
)

for dataset in "${datasets[@]}"; do
    echo "Evaluate 8b Embeddings for ${dataset}"
    python evaluate_readmission.py \
        --embeddings-train-path artifacts/${dataset}8b/train/qwen3_embeddings.feather \
        --embeddings-test-path artifacts/${dataset}8b/test/qwen3_embeddings.feather \
        --labels-train-path artifacts/${dataset}8b/train/qwen3_embeddings_labels.feather \
        --labels-test-path artifacts/${dataset}8b/test/qwen3_embeddings_labels.feather \
        --max-iter 1000 \
        --cv-folds 5 \
        --model-output-path models/${dataset}MODEL.pkl
done
