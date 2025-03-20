from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict

from causallearn.graph.GeneralGraph import GeneralGraph
from src.graph_aux import get_edge_adjacency_matrix, get_graph_skeleton
from src.logging_config import setup_logging
from src.metrics import Metrics
from src.visualization import Plotter


class CausalDiscoveryAlgorithm(ABC):
    def __init__(self, config_params):
        self.config_params = config_params

        # Not all graphs may have this. I'll leave unset
        self.est_adj: np.ndarray = np.ndarray([])
        self.est_edge_adj: np.ndarray = np.ndarray([])

        self.est_graph: GeneralGraph = GeneralGraph([])
        self.est_graph_skeleton: GeneralGraph = GeneralGraph([])
        self.is_trained = False

    @abstractmethod
    def train(self, data: np.ndarray) -> None:
        """Learns causal structure from data"""
        pass

    def _set_auxiliary_results(self):
        if self.est_graph == GeneralGraph([]):
            raise RuntimeError("Estimated graph is empty.")

        self.est_edge_adj = get_edge_adjacency_matrix(graph=self.est_graph)
        self.est_graph_skeleton = get_graph_skeleton(graph=self.est_graph)
        self.is_trained = True

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
