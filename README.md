# Causal Discovery Algorithm Comparison

## Overview
This repository allows to run multiple causal discovery algorithms at once for some dataset. Currently the following algorithms are available:
- causal-learn
  - PC, FCI, ES, GES, GRASP, DirectLiNGAM, ICALiNGAM,
- gCastle
  - NOTEARS, DAG_GNN, GRANDAG, CORL

## Requirements
```
# Create venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt 

```

## How to use

### Algorithm configuration
In the file `configs/algorithms.yaml`, add and configure an algorithm. For example:
```
<config_tag>:
    algorithm: <algorithm_name>
    <param1_key>: <parameter1_value>
    <param2_key>: <parameter2_value>
    ...
```
`<config_tag>` can be any name you wish. We will use this name to select the configuration when running the script. `<algorithm_name>` should be chosen between the available algorithms. Please see file `main.py` to see which algorithms are available. Then, feel free to specify parameters for the algorithm. The causal-learn and gCastle libraries show which parameters are available for each algorithm. An example of the algorithm with default values is set in the config file.

### Data configuration

In the file `configs/data.yaml`, specify the path to your training data and ground truth adjacency matrix. This adjacency matrix will be used to calculate the confusion matrix of arrows and edges.

```
<dataset_config_name>:
    train_fpath: .path/to/train.csv
    true_adj_fpath: ./path/to/adj_matrix.csv
```

### Dagshub configuration (optional)
Then, specify your enviroment variables if you wish to log mlflow metrics in dagshub. Create a file `.env` in the root folder and specify:

```
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo-name>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<secret-access-key>
```

If you do not wish to send mlflow metrics to dagshub, leave these empty. A local folder `/mlruns` will be created with the logs.

### Running
To run the script for a chosen configuration and dataset:

```
python3 main.py --algorithm <config_tag> --data <dataset_config_name> 
```
The file `scripts/run_experiments.bash` has an bash script example of how to run multiple experiments in sequence.

### Output
Expect metrics and plots to be logged at `/results`. If you configured dagsflow, results should be there. If not, mlflow logs will be at `mlruns`. To view them locally, execute:
```
mlflow ui
```
