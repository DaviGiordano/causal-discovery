from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.SHD import SHD
from src.graph_aux import get_graph_skeleton
from typing import Dict


class Metrics:
    def __init__(
        self,
        true_graph: GeneralGraph,
        est_graph: GeneralGraph,
        training_time: float,
    ) -> None:
        self.true_graph = true_graph
        self.est_graph = est_graph
        self.training_time = training_time
        self._validate_graphs()

        # Calculate all metrics
        self.result_metrics = {
            "adjacency": self._compute_adjacency_metrics(),
            "arrow": self._compute_arrow_metrics(),
            "arrow_ce": self._compute_arrow_ce_metrics(),
            "shd": self._compute_shd(),
            "skeleton_shd": self._compute_skeleton_shd(),
            "training_time": self.training_time,
        }

    def _validate_graphs(self):
        true_nodes = set(node.get_name() for node in self.true_graph.get_nodes())
        est_nodes = set(node.get_name() for node in self.est_graph.get_nodes())
        if true_nodes != est_nodes:
            raise ValueError(
                f"Graphs have different nodes.\nTrue: {true_nodes}\nEstimated: {est_nodes}"
            )

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision == -1 or recall == -1:
            return -1
        return round(2 * (precision * recall) / (precision + recall), 2)

    def _compute_adjacency_metrics(self) -> Dict:
        """Compute metrics for adjacency comparison."""
        adj = AdjacencyConfusion(self.true_graph, self.est_graph)

        try:
            precision = round(adj.get_adj_precision(), 2)
        except ZeroDivisionError:
            precision = -1
        try:
            recall = round(adj.get_adj_recall(), 2)
        except ZeroDivisionError:
            recall = -1

        return {
            "confusion_matrix": [
                [adj.get_adj_tp(), adj.get_adj_fp()],
                [adj.get_adj_fn(), adj.get_adj_tn()],
            ],
            "precision": precision,
            "recall": recall,
            "f1": self._calculate_f1(precision, recall),
        }

    def _compute_arrow_metrics(self) -> Dict:
        """Compute metrics for directed edge comparison."""
        arrow = ArrowConfusion(self.true_graph, self.est_graph)

        try:
            precision = round(arrow.get_arrows_precision(), 2)
        except ZeroDivisionError:
            precision = -1
        try:
            recall = round(arrow.get_arrows_recall(), 2)
        except ZeroDivisionError:
            recall = -1

        return {
            "confusion_matrix": [
                [arrow.get_arrows_tp(), arrow.get_arrows_fp()],
                [arrow.get_arrows_fn(), arrow.get_arrows_tn()],
            ],
            "precision": precision,
            "recall": recall,
            "f1": self._calculate_f1(precision, recall),
        }

    def _compute_arrow_ce_metrics(self) -> Dict:
        """Compute metrics for ce directed edge. (?)"""
        arrow = ArrowConfusion(self.true_graph, self.est_graph)

        try:
            precision = round(arrow.get_arrows_precision_ce(), 2)
        except ZeroDivisionError:
            precision = -1
        try:
            recall = round(arrow.get_arrows_recall_ce(), 2)
        except ZeroDivisionError:
            recall = -1

        return {
            "confusion_matrix": [
                [arrow.get_arrows_tp_ce(), arrow.get_arrows_fp_ce()],
                [arrow.get_arrows_fn_ce(), arrow.get_arrows_tn_ce()],
            ],
            "precision": precision,
            "recall": recall,
            "f1": self._calculate_f1(precision, recall),
        }

    def _compute_shd(self) -> float:
        """Compute SHD distance between two graphs"""
        return SHD(self.true_graph, self.est_graph).get_shd()

    def _compute_skeleton_shd(self) -> float:
        """Compute SHD distance between the skeleton of two graphs"""
        return SHD(
            get_graph_skeleton(self.true_graph),
            get_graph_skeleton(self.est_graph),
        ).get_shd()

    def get_result_metrics(self) -> Dict:
        """Return all metrics."""
        return self.result_metrics
