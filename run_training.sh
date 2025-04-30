#!/bin/bash

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow UI server in the background
mlflow ui --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!

# Run the training container with proper volume mounting
docker run --gpus all \
  -v "$(pwd)":/app \
  -v "$(pwd)/mlruns":/app/mlruns \
  --memory=8g --memory-swap=16g \
  dnn-training python train.py "$@"

# Cleanup: Kill MLflow UI server
kill $MLFLOW_PID

echo "Training completed. MLflow UI has been closed." 