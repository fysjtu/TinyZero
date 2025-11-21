#!/bin/bash
# Test script for evaluating trained models

# Set environment variables
export PYTHONPATH=/root/TinyZero:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Model and data paths
MODEL_CHECKPOINT="/root/autodl-tmp/checkpoints/actor/global_step_100"
TEST_DATA_PATH="/root/TinyZero/data/test.parquet"

# Check if model exists
if [ ! -d "$MODEL_CHECKPOINT" ]; then
    echo "Model checkpoint not found at $MODEL_CHECKPOINT"
    echo "Available checkpoints:"
    ls -la /root/autodl-tmp/checkpoints/actor/
    exit 1
fi

# Run evaluation
echo "Running model evaluation..."
echo "Model checkpoint: $MODEL_CHECKPOINT"
echo "Test data: $TEST_DATA_PATH"

python3 /root/TinyZero/scripts/test_model.py

echo "Evaluation completed!"