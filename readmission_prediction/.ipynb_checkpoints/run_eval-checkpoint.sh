#!/bin/bash

# run_eval.sh

datasets=(
    "_name_address_age_visitdates_gender_"
    "_name_address_age_visitdates_"
    "_name_address_age_gender_"
    "_name_address_age_"
    "_name_address_visitdates_gender_"
    "_name_address_visitdates_"
    "_name_address_gender_"
    "_name_address_"
    "_name_age_visitdates_gender_"
    "_name_age_visitdates_"
    "_name_age_gender_"
    "_name_age_"
    "_name_visitdates_gender_"
    "_name_visitdates_"
    "_name_gender_"
    "_name_"
    "_address_age_visitdates_gender_"
    "_address_age_visitdates_"
    "_address_age_gender_"
    "_address_age_"
    "_address_visitdates_gender_"
    "_address_visitdates_"
    "_address_gender_"
    "_address_"
    "_age_visitdates_gender_"
    "_age_visitdates_"
    "_age_gender_"
    "_age_"
    "_visitdates_gender_"
    "_visitdates_"
    "_gender_"
    "_"
    "_name_address_age_visitdates_gender_race_"
    "_name_address_age_visitdates_race_"
    "_name_address_age_gender_race_"
    "_name_address_age_race_"
    "_name_address_visitdates_gender_race_"
    "_name_address_visitdates_race_"
    "_name_address_gender_race_"
    "_name_address_race_"
    "_name_age_visitdates_gender_race_"
    "_name_age_visitdates_race_"
    "_name_age_gender_race_"
    "_name_age_race_"
    "_name_visitdates_gender_race_"
    "_name_visitdates_race_"
    "_name_gender_race_"
    "_name_race_"
    "_address_age_visitdates_gender_race_"
    "_address_age_visitdates_race_"
    "_address_age_gender_race_"
    "_address_age_race_"
    "_address_visitdates_gender_race_"
    "_address_visitdates_race_"
    "_address_gender_race_"
    "_address_race_"
    "_age_visitdates_gender_race_"
    "_age_visitdates_race_"
    "_age_gender_race_"
    "_age_race_"
    "_visitdates_gender_race_"
    "_visitdates_race_"
    "_gender_race_"
    "_race_"
)

for dataset in "${datasets[@]}"; do
    echo "Evaluate 8b Embeddings for ${dataset} Train Split"
    python evaluate_readmission.py \
        --embeddings-train-path artifacts/${dataset}8b/train/qwen3_embeddings.feather \
        --embeddings-test-path artifacts/${dataset}8b/test/qwen3_embeddings.feather \
        --labels-train-path artifacts/${dataset}8b/train/qwen3_embeddings_labels.feather \
        --labels-test-path artifacts/${dataset}8b/test/qwen3_embeddings_labels.feather \
        --max-iter 1000 \
        --cv-folds 5 \
        --output-path outputs/${dataset}8b_cv5.json
done
