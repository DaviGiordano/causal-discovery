import pathlib
import time
from src.metrics import Metrics
from src.causal_discovery.causallearn_algorithms import (
    PCAlgorithm,
    FCIAlgorithm,
    GESAlgorithm,
    ExactSearchAlgorithm,
    ICALiNGAMAlgorithm,
    DirectLiNGAMAlgorithm,
    GRaSPAlgorithm,
)

from src.causal_discovery.castle_algorithms import (
    NOTEARSAlgorithm,
    DAGGNNAlgorithm,
    CORLAlgorithm,
    GraNDAGAlgorithm,
)
from src.load_parse import load_csv, load_yaml, parse_arguments
from src.graph_aux import dag_adj_to_graph
from src.visualization import Plotter
from src.json_logger import log_experiment_results
from src.logging_config import setup_logging
from src.mlflow_logger import MLflowLogger

ALL_ALGORITHMS_CONFIGS = "./configs/algorithms.yaml"
ALL_DATA_CONFIGS = "./configs/data.yaml"


def get_discovery_algorithm(algorithm_params: dict):
    config_params = algorithm_params.copy()
    algorithm = config_params.pop("algorithm")
    if algorithm == "pc":
        return PCAlgorithm(config_params)
    elif algorithm == "fci":
        return FCIAlgorithm(config_params)
    elif algorithm == "ges":
        return GESAlgorithm(config_params)
    elif algorithm == "es":
        return ExactSearchAlgorithm(config_params)
    elif algorithm == "icalingam":
        return ICALiNGAMAlgorithm(config_params)
    elif algorithm == "directlingam":
        return DirectLiNGAMAlgorithm(config_params)
    elif algorithm == "grasp":
        return GRaSPAlgorithm(config_params)
    elif algorithm == "notears":
        return NOTEARSAlgorithm(config_params)
    elif algorithm == "dag_gnn":
        return DAGGNNAlgorithm(config_params)
    elif algorithm == "corl":
        return CORLAlgorithm(config_params)
    elif algorithm == "grandag":
        return GraNDAGAlgorithm(config_params)
    else:
        raise NotImplementedError(f"{algorithm} was not yet implemented")


def main():
    # Setup logging
    setup_logging()

    # Parse args
    args = parse_arguments()

    # Setup MLflow logging
    mlflow_logger = MLflowLogger(experiment_name=args.dataset_config)

    # Load data and ground truth adj matrix
    data_params = load_yaml(ALL_DATA_CONFIGS)[args.dataset_config]
    data = load_csv(data_params["train_fpath"])
    true_adj = load_csv(data_params["true_adj_fpath"])
    true_graph = dag_adj_to_graph(true_adj)

    # Load selected model
    algorithm_params = load_yaml(ALL_ALGORITHMS_CONFIGS)[args.algorithm_config]
    model = get_discovery_algorithm(algorithm_params)

    # Train model to discover causal structure and measure time
    start_time = time.time()
    model.train(data)
    training_time = time.time() - start_time
    est_graph = model.est_graph

    # Evaluate and get metrics
    metrics = Metrics(true_graph, est_graph)
    metrics_results = metrics.get_result_metrics()
    metrics_results["train_time"] = round(training_time, 2)

    # Setup output directory
    output_dir = pathlib.Path(f"results/{args.dataset_config}/{args.algorithm_config}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log experiment parameters and results to JSON
    log_experiment_results(
        output_dir=output_dir,
        params=algorithm_params,
        metrics=metrics_results,
    )

    # Generate and save plots
    plotter = Plotter()
    plotter.plot_confusion_comparison(
        metrics_data=metrics.get_result_metrics(),
        title=f"Confusion Matrices - {args.algorithm_config} - {args.dataset_config}",
        fpath=f"{output_dir}/confusion_matrices.png",
    )
    plotter.plot_graph(
        title=f"True Graph - {args.dataset_config}",
        graph=true_graph,
        fpath=f"{output_dir}/true_graph.png",
    )
    plotter.plot_graph(
        title=f"Estimated Graph - {args.algorithm_config} - {args.dataset_config}",
        graph=est_graph,
        fpath=f"{output_dir}/est_graph.png",
    )
    plotter.plot_graph_comparison(
        graph1=true_graph,
        graph2=est_graph,
        fpath=f"{output_dir}/graph_comparison.png",
        title=f"Graph Comparison - {args.algorithm_config} - {args.dataset_config}",
    )

    # Log to MLflow
    mlflow_logger.log_run(
        run_name=args.algorithm_config,
        params=algorithm_params,
        metrics=metrics_results,
        artifacts_dir=output_dir,
    )


if __name__ == "__main__":

    main()
