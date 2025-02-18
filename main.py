from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from omegaconf import OmegaConf
from causallearn.utils.GraphUtils import GraphUtils

import pandas as pd
from utils.logging import setup_logging
from utils.plot import (
    plot_confusion_comparison,
    plot_graph_comparison,
)
import logging
from utils.algorithm import run_causal_discovery
import argparse
import mlflow
from utils.metrics import get_graph_confusion


def load_data_true_g(data_tag):
    data_fpath = f"./data/csuite/{data_tag}/train.csv"
    true_g_fpath = f"./data/csuite/{data_tag}/tetrad.txt"
    data = pd.read_csv(data_fpath, header=None).to_numpy()
    true_g = txt2generalgraph(true_g_fpath)
    return data, true_g


def load_config(algorithm_tag):
    return OmegaConf.load(f"config/{algorithm_tag}.yaml")


def main(algorithm_tag, data_tag):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(data_tag)
    with mlflow.start_run(run_name=algorithm_tag):
        logger, log_dir = setup_logging(algorithm_tag, data_tag)
        data, true_g = load_data_true_g(data_tag)
        config = load_config(algorithm_tag)
        graph_utils = GraphUtils()
        mlflow.log_params(config)
        logger.info(f"True graph nodes: {graph_utils.graph_string(true_g)}")
        est_g = run_causal_discovery(algorithm_tag, data, config)
        logger.info(f"Estimated graph nodes: {graph_utils.graph_string(est_g)}")
        metrics = {}
        for graph_type in ["adj", "arrow", "arrow_ce"]:
            cm, precision, recall = get_graph_confusion(graph_type, true_g, est_g)
            metrics.update(
                {
                    f"{graph_type}_tp": cm[0][0],
                    f"{graph_type}_fp": cm[0][1],
                    f"{graph_type}_fn": cm[1][0],
                    f"{graph_type}_tn": cm[1][1],
                    f"{graph_type}_precision": precision,
                    f"{graph_type}_recall": recall,
                }
            )

        mlflow.log_metrics(metrics)

        # 5. Log artifacts (confusion matrix and graph plots)
        fig_conf = plot_confusion_comparison(
            true_g, est_g, fpath=str(log_dir / "confusion.png")
        )
        fig_graphs = plot_graph_comparison(
            true_g, est_g, fpath=str(log_dir / "graphs.png")
        )

        mlflow.log_artifact(str(log_dir / "confusion.png"))
        mlflow.log_artifact(str(log_dir / "graphs.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal Discovery Main Script")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="The algorithm to use for causal discovery",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="The data to run the algorithm on"
    )

    args = parser.parse_args()

    main(args.algorithm, args.data)
