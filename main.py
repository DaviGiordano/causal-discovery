import os
import pathlib
import time
import logging
from dotenv import load_dotenv
from src.metrics import Metrics
from src.causal_discovery.causallearn_algorithms import (
    PCAlgorithm,
    FCIAlgorithm,
    GESAlgorithm,
    ExactSearchAlgorithm,
    ICALiNGAMAlgorithm,
    DirectLiNGAMAlgorithm,
    GRaSPAlgorithm,
    BossAlgorithm,
)

# from src.causal_discovery.castle_algorithms import (
#     NOTEARSAlgorithm,
#     DAGGNNAlgorithm,
#     CORLAlgorithm,
#     GraNDAGAlgorithm,
# )
from src.load_parse import load_csv, load_yaml, parse_arguments
from src.graph_aux import dag_adj_to_graph
from src.visualization import Plotter
from src.json_logger import log_experiment_results
from src.logging_config import setup_logging
from src.mlflow_logger import MLflowLogger

ALL_ALGORITHMS_CONFIGS = "./configs/algorithms.yaml"
ALL_DATA_CONFIGS = "./configs/dataset.yaml"


def get_discovery_algorithm(algorithm_name: str, algorithm_params: dict):
    if algorithm_name == "pc":
        return PCAlgorithm(algorithm_params)
    elif algorithm_name == "fci":
        return FCIAlgorithm(algorithm_params)
    elif algorithm_name == "ges":
        return GESAlgorithm(algorithm_params)
    elif algorithm_name == "es":
        return ExactSearchAlgorithm(algorithm_params)
    elif algorithm_name == "icalingam":
        return ICALiNGAMAlgorithm(algorithm_params)
    elif algorithm_name == "directlingam":
        return DirectLiNGAMAlgorithm(algorithm_params)
    elif algorithm_name == "grasp":
        return GRaSPAlgorithm(algorithm_params)
    elif algorithm_name == "boss":
        return BossAlgorithm(algorithm_params)
    # elif algorithm_name == "notears":
    #     return NOTEARSAlgorithm(algorithm_params)
    # elif algorithm_name == "dag_gnn":
    #     return DAGGNNAlgorithm(algorithm_params)
    # elif algorithm_name == "corl":
    #     return CORLAlgorithm(algorithm_params)
    # elif algorithm_name == "grandag":
    #     return GraNDAGAlgorithm(algorithm_params)
    else:
        raise NotImplementedError(f"{algorithm_name} was not yet implemented")


def main():
    # Setup logging
    load_dotenv(".env")
    args = parse_arguments()

    # Setup output directory
    output_dir = pathlib.Path(f"results/{args.dataset_tag}/{args.algorithm_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging_fpath = str(output_dir / "output.log")
    setup_logging(logging_fpath)

    # Capture warnings as log messages
    logging.captureWarnings(True)

    try:

        # Setup MLflow logging
        mlflow_logger = MLflowLogger(
            experiment_name=args.dataset_tag,
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )

        # Load data and ground truth adj matrix
        data_params = load_yaml(ALL_DATA_CONFIGS)[args.dataset_tag]
        data = load_csv(data_params["train_fpath"])
        true_adj = load_csv(data_params["true_adj_fpath"])
        true_graph = dag_adj_to_graph(true_adj)

        # Load selected model
        algorithm_config = load_yaml(ALL_ALGORITHMS_CONFIGS)[args.algorithm_tag]
        model = get_discovery_algorithm(
            algorithm_config["algorithm_name"],
            algorithm_config["algorithm_params"],
        )

        # Train model to discover causal structure and measure time
        start_time = time.time()
        model.train(data)
        training_time = time.time() - start_time
        est_graph = model.est_graph

        # Evaluate and get metrics
        metrics = Metrics(true_graph, est_graph, training_time)
        metrics_results = metrics.get_result_metrics()

        # Log experiment parameters and results to JSON
        log_experiment_results(
            output_dir=output_dir,
            params=algorithm_config["algorithm_params"],
            metrics=metrics_results,
        )

        # Generate and save plots
        plotter = Plotter()
        plotter.plot_confusion_comparison(
            metrics_data=metrics.get_result_metrics(),
            title=f"Confusion Matrices - {args.algorithm_tag} - {args.dataset_tag}",
            fpath=f"{output_dir}/{args.algorithm_tag}_{args.dataset_tag}_confusion_matrices.png",
        )
        plotter.plot_graph(
            title=f"True Graph - {args.dataset_tag}",
            graph=true_graph,
            fpath=f"{output_dir}/true_graph.png",
        )
        plotter.plot_graph(
            title=f"Estimated Graph - {args.algorithm_tag} - {args.dataset_tag}",
            graph=est_graph,
            fpath=f"{output_dir}/{args.algorithm_tag}_{args.dataset_tag}_est_graph.png",
        )
        plotter.plot_graph_comparison(
            graph1=true_graph,
            graph2=est_graph,
            fpath=f"{output_dir}/{args.algorithm_tag}_{args.dataset_tag}_graph_comparison.png",
            title=f"Graph Comparison - {args.algorithm_tag} - {args.dataset_tag}",
        )

        # Log to MLflow
        mlflow_logger.log_run(
            run_name=args.algorithm_tag,
            dataset_name=args.dataset_tag,
            params=algorithm_config["algorithm_params"],
            metrics=metrics_results,
            artifacts_dir=output_dir,
        )
    except Exception:
        logging.exception("Unhandled exception.")  # Logs full traceback


if __name__ == "__main__":

    main()
