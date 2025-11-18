#!/bin/bash

#./run_embed_gender.sh


INPUT_DIR="gender_swapping/data"

OUTPUT_BASE="gender_swapping/artifacts"

echo "Starting embedding computation for all CSVs in $INPUT_DIR..."

for csv_file in "$INPUT_DIR"/*.csv; do

    filename=$(basename "$csv_file" .csv)

    output_dir="${OUTPUT_BASE}/${filename}_8b"

    mkdir -p "$output_dir"

    echo "Computing embeddings for $filename..."
    python3 precompute_embeddings.py \
        --csv-path "$csv_file" \
        --output-dir "$output_dir" \
        --model "8b"
done

echo "All embeddings computed."

