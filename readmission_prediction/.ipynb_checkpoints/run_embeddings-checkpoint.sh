#!/bin/bash

# run_embeddings.sh


compute_embeddings() {
    local name=$1
    echo "Compute 8b Embeddings for ${name} Train Split"
    python3 precompute_embeddings.py --csv-path data/${name}/train.csv --output-dir artifacts/${name}8b/train --model 8b
    echo "Compute 8b Embeddings for ${name} Test Split"
    python3 precompute_embeddings.py --csv-path data/${name}/test.csv --output-dir artifacts/${name}8b/test --model 8b
}

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

)

for dataset in "${datasets[@]}"; do
    compute_embeddings "$dataset"
done


