import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict

from causallearn.graph.GeneralGraph import GeneralGraph
from src.logging_config import setup_logging
from src.metrics import Metrics
from src.visualization import Plotter


class CausalDiscoveryAlgorithm(ABC):
    def __init__(self, config_params):
        self.config_params = config_params
        self.true_adj = None
        self.true_graph = None
        self.est_adj = None
        self.est_graph: GeneralGraph | None
        self.metrics: Metrics | None = None

    @abstractmethod
    def train(self, data: np.ndarray) -> None:
        """Learns causal structure from data"""
        pass

    # def evaluate_graph(self) -> Dict:
    #     """Evaluate the estimated graph against the true graph.

    #     Returns:
    #         Dict: Dictionary containing adjacency, arrow, and arrow_ce metrics
    #     """
    #     if self.true_graph is None or self.est_graph is None:
    #         logger.error("Cannot evaluate: true_graph or est_graph is None")
    #         return {}

    #     try:
    #         self.metrics = Metrics(self.true_graph, self.est_graph)
    #         result_metrics = self.metrics.get_result_metrics()

    #         # Log the results
    #         logger.info("Graph evaluation results:")
    #         for metric_type, values in result_metrics.items():
    #             logger.info(f"{metric_type}:")
    #             logger.info(f"  Precision: {values['precision']}")
    #             logger.info(f"  Recall: {values['recall']}")
    #             logger.info(f"  F1: {values['f1']}")

    #         return result_metrics

    #     except Exception as e:
    #         logger.error(f"Error during graph evaluation: {str(e)}")
    #         return {}
