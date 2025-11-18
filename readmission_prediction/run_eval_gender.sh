#!/bin/bash

# run_eval_gender.sh

BASE_DIR="readmission_prediction/gender_swapping"
ARTIFACTS_DIR="${BASE_DIR}/artifacts"
OUTPUT_FULL="${BASE_DIR}/outputs/full_info"
OUTPUT_ANON="${BASE_DIR}/outputs/anonymized"

MODEL_FULL="readmission_prediction/models/_name_address_age_visitdates_gender_race_MODEL.pkl"
MODEL_ANON="readmission_prediction/models/_MODEL.pkl"

echo "Starting evaluation for all *_8b datasets in ${ARTIFACTS_DIR}..."

for embeddings_dir in "${ARTIFACTS_DIR}"/*_8b; do

    dataset=$(basename "$embeddings_dir")

    echo "Evaluating 8b embeddings of ${dataset} for full_info model..."
    python3 evaluate_readmission.py \
        --model-input-path "$MODEL_FULL" \
        --embeddings-test-path "${embeddings_dir}/qwen3_embeddings.feather" \
        --labels-test-path "${embeddings_dir}/qwen3_embeddings_labels.feather" \
        --output-path "${OUTPUT_FULL}/${dataset}_full.json"

    echo "Evaluating 8b embeddings of ${dataset} for anonymized model..."
    python3 evaluate_readmission.py \
        --model-input-path "$MODEL_ANON" \
        --embeddings-test-path "${embeddings_dir}/qwen3_embeddings.feather" \
        --labels-test-path "${embeddings_dir}/qwen3_embeddings_labels.feather" \
        --output-path "${OUTPUT_ANON}/${dataset}_anon.json"
done

echo "All evaluations completed."
