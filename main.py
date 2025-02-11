from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from omegaconf import OmegaConf
import pandas as pd
from utils.logging import setup_logging
from utils.plot import (
    plot_confusion_comparison,
    plot_graph_comparison,
)
import logging
from utils.algorithm import run_causal_discovery
import argparse
import logging


def load_data_true_g(data_tag):
    data_fpath = f"./data/csuite/{data_tag}/train.csv"
    true_g_fpath = f"./data/csuite/{data_tag}/tetrad.txt"
    data = pd.read_csv(data_fpath, header=None).to_numpy()
    true_g = txt2generalgraph(true_g_fpath)
    return data, true_g


def load_config(algorithm_tag):
    return OmegaConf.load(f"config/{algorithm_tag}.yaml")


def main(algorithm_tag, data_tag):

    logger, log_dir = setup_logging(algorithm_tag, data_tag)
    data, true_g = load_data_true_g(data_tag)
    config = load_config(algorithm_tag)

    est_g = run_causal_discovery(algorithm_tag, data, config)

    fig_conf = plot_confusion_comparison(
        true_g, est_g, fpath=str(log_dir / "confusion.png")
    )
    fig_graphs = plot_graph_comparison(true_g, est_g, fpath=str(log_dir / "graphs.png"))


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

    logging.basicConfig(level=logging.INFO)

    main(args.algorithm, args.data)
