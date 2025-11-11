
# Guide/useful commands
Most of these commands are tested on Windows in the Command Prompt unless stated otherwise
## Training
Using experiment interface
```
docker run --gpus all -v "%cd%":/app -v "%cd%/mlruns":/app/mlruns --memory=8g --memory-swap=16g dnn-training python experiment_runner.py experiment.json
```

## Analysis
To start a jupyter notebook in Docker:
```
docker run -it --rm -p 8888:8888 --gpus all  -v "notebooks:/app/notebooks"  my-analysis-app
```
Then:
1. Copy the `http://127[...]` link
2. Open a jupyter notebook in your notebooks folder
3. Select kernel -> "Select another kernel"  -> "Existing Jupyter Server" -> paste link

To rebuild the jupyter notebook Docker container:
```
docker build -t my-analysis-app -f Dockerfile.analysis . 
```

To quickly explore using mlflow locally run 
```
mlflow ui
```


# Tasks

## TODO

## Completed

# Sources

https://github.com/agaldran/cost_sensitive_loss_classification 
