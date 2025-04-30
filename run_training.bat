@echo off

:: Create mlruns directory if it doesn't exist
if not exist mlruns mkdir mlruns

:: Start MLflow UI server in the background using conda environment
start cmd /k "conda activate dnn-mode-connectivity && mlflow ui --host 0.0.0.0 --port 5000"

:: Run the training container with proper volume mounting
docker run --gpus all ^
  -v %cd%:/app ^
  -v %cd%/mlruns:/app/mlruns ^
  --memory=8g --memory-swap=16g ^
  dnn-training python train.py %*

:: Keep the window open to show training progress
echo Training in progress. MLflow UI is available at http://localhost:5000
echo Press Ctrl+C to stop training and close MLflow UI
pause 