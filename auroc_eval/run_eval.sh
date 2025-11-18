#!/bin/bash

#run_eval.sh


for i in {5..95..5}; do
  echo "Evaluate 8b Embeddings for test$i"
  python evaluate_readmission.py \
    --embeddings-train-path /splits/test$i/artifacts/train_8b/qwen3_embeddings.feather \
    --embeddings-test-path splits/test$i/artifacts/test_8b/qwen3_embeddings.feather \
    --labels-train-path splits/test$i/artifacts/train_8b/qwen3_embeddings_labels.feather \
    --labels-test-path splits/test$i/artifacts/test_8b/qwen3_embeddings_labels.feather \
    --max-iter 1000 \
    --cv-folds 5 \
    --output-path outputs/test${i}_cv5.json
done