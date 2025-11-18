#!/bin/bash

# run_embeddings.sh

# Helper function
compute_embeddings() {
    local name=$1
    local label=$2

    # Choose correct label column
    if [ "$label" == "age" ]; then
        label_column="actual_age"
    else
        label_column="$label"
    fi

    echo "Compute 8b Embeddings for ${label}/${name} Train Split"
    python3 ${label}/precompute_embeddings.py --csv-path ${label}/data/${name}/train.csv --output-dir ${label}/artifacts/${name}_8b/train --model 8b --label-column ${label_column}

    echo "Compute 8b Embeddings for ${label}/${name} Test Split"
    python3 ${label}/precompute_embeddings.py --csv-path ${label}/data/${name}/test.csv --output-dir ${label}/artifacts/${name}_8b/test --model 8b --label-column ${label_column}
}

# Define groups
labels=("age" "race" "gender")
variants=("given" "no")
contexts=("everything" "nothing")

# Loop through label types (age, race, gender)
for label in "${labels[@]}"; do
    for v in "${variants[@]}"; do
        for c in "${contexts[@]}"; do
            dataset="${v}_${label}_${c}"
            compute_embeddings "$dataset" "$label"
        done
    done
done

