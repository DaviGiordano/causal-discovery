from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.SHD import SHD
from src.graph_aux import get_graph_skeleton
from typing import Dict
from statistics import mean


class Metrics:
    def __init__(
        self,
        training_time: float,
        true_graph: GeneralGraph,
        est_graph: GeneralGraph,
        est_edges_dict: dict = None,
        true_edges_dict: dict = None,
        edge_probabilities: dict = None,
    ) -> None:
        self.true_graph = true_graph
        self.est_graph = est_graph
        self.training_time = training_time
        self._validate_graphs()

        self.est_edges_dict = est_edges_dict
        self.true_edges_dict = true_edges_dict
        self.edge_probabilities = edge_probabilities

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

    def _compute_average_frequency(self) -> float:
        """Compute average frequency of the chosen edges, including absence of edge."""
        frequencies = []
        for edge_key, chosen_edge in self.est_edges_dict.items():
            frequencies.append(self.edge_probabilities[edge_key][chosen_edge])
        if not frequencies:
            return 0
        return mean(frequencies)

    def _compute_average_edge_frequency(self) -> float:
        """Compute average frequency of the chosen edges, excluding absence of edge"""
        frequencies = []
        for edge_key, chosen_edge in self.est_edges_dict.items():
            if chosen_edge != "no_edge":
                frequencies.append(self.edge_probabilities[edge_key][chosen_edge])
        if not frequencies:
            return 0
        return mean(frequencies)

    def _compute_brier_score(self) -> float:
        """
        Compute the Brier score for the graph predictions.
        Each edge is a multiclass problem between ['no_edge', 'source->target', 'target->source']
        See "Original Definition by Brier" at en.wikipedia.org/wiki/Brier_score

        Brier score measures the accuracy of probabilistic predictions, calculated as
        the mean squared difference between predicted probabilities and actual outcomes.
        Lower values indicate better calibration of probabilities (0 is perfect).

        Returns:
            float: The calculated Brier score
        """
        if not self.edge_probabilities or not self.true_edges_dict:
            return 0.0

        squared_errors = []
        # For all edges
        for edge_key, prob_dict in self.edge_probabilities.items():
            true_edge_type = self.true_edges_dict.get(edge_key, "no_edge")

            # For each edge class
            for edge_type, probability in prob_dict.items():

                if edge_type == "edge":  # Skip 'edge'
                    continue

                # Correct edge class is 1, incorrect edge classes are 0
                encoded_true_edge_type = 1.0 if edge_type == true_edge_type else 0.0

                squared_error = (probability - encoded_true_edge_type) ** 2
                squared_errors.append(squared_error)

        if not squared_errors:
            return 0.0

        return sum(squared_errors) / len(squared_errors)

    def _compute_expected_calibration_error(self, num_bins=10) -> float:
        """
        Compute the Expected Calibration Error (ECE) for the graph predictions.
        Considers only the probability of the chosen edge

        Args:
            num_bins (int): Number of bins to divide the probability range [0,1] into.

        Returns:
            float: The calculated Expected Calibration Error
        """
        if (
            not self.edge_probabilities
            or not self.true_edges_dict
            or not self.est_edges_dict
        ):
            return 0.0

        bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
        bins = {i: {"correct": 0, "total": 0, "sum_prob": 0.0} for i in range(num_bins)}

        # Sum accuracy and probabilities for the edges (considers only chosen edge)
        for edge_key, chosen_edge_type in self.est_edges_dict.items():

            true_edge_type = self.true_edges_dict.get(edge_key, "no_edge")
            probability = self.edge_probabilities[edge_key].get(chosen_edge_type, 0.0)
            bin_index = min(int(probability * num_bins), num_bins - 1)

            bins[bin_index]["total"] += 1
            bins[bin_index]["sum_prob"] += probability

            if chosen_edge_type == true_edge_type:
                bins[bin_index]["correct"] += 1

        # Calculate ECE
        ece = 0.0
        total_predictions = sum(bin_data["total"] for bin_data in bins.values())

        if total_predictions == 0:
            return -1

        for bin_idx, bin_data in bins.items():
            if bin_data["total"] > 0:
                avg_predicted_prob = bin_data["sum_prob"] / bin_data["total"]
                accuracy = bin_data["correct"] / bin_data["total"]
                bin_weight = bin_data["total"] / total_predictions

                ece += bin_weight * abs(avg_predicted_prob - accuracy)

        return ece

    def get_result_metrics(self) -> Dict:
        result_metrics = {
            "adjacency": self._compute_adjacency_metrics(),
            "arrow": self._compute_arrow_metrics(),
            "arrow_ce": self._compute_arrow_ce_metrics(),
            "shd": self._compute_shd(),
            "skeleton_shd": self._compute_skeleton_shd(),
            "training_time": self.training_time,
            "average_frequency": (
                self._compute_average_frequency()
                if (self.est_edges_dict and self.edge_probabilities)
                else -1
            ),
            "average_edge_frequency": (
                self._compute_average_edge_frequency()
                if (self.est_edges_dict and self.edge_probabilities)
                else -1
            ),
            "brier": (
                self._compute_brier_score()
                if (self.est_edges_dict and self.edge_probabilities)
                else -1
            ),
            "ece": (
                self._compute_expected_calibration_error()
                if (self.est_edges_dict and self.edge_probabilities)
                else -1
            ),
        }
        """Return all metrics."""
        return result_metrics
