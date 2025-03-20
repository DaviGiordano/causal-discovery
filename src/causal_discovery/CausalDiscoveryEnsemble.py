import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from src.algorithm_choice import get_discovery_algorithm
from src.causal_discovery.CausalDiscoveryAlgorithm import CausalDiscoveryAlgorithm
from src.graph_aux import dag_adj_to_graph, to_pydot_label_edges
from src.load_parse import load_csv, load_yaml
from causallearn.graph.GraphNode import GraphNode

from src.visualization import Plotter

logger = logging.getLogger(__name__)


class CausalDiscoveryEnsemble:
    def __init__(self, algorithms: List[CausalDiscoveryAlgorithm]):
        self.algorithms = algorithms

        self.is_ensemble_trained: bool = False
        self.avg_edge_adj: np.ndarray = np.array([])
        self.est_skeleton_union: GeneralGraph = GeneralGraph([])
        self.edge_frequency: Dict[Tuple[int, int], str] = {}
        self.est_directed_union: GeneralGraph = GeneralGraph([])
        self.directed_frequency: Dict[Tuple[int, int], str] = {}

    def train(self, data: np.ndarray):
        for algo in self.algorithms:
            algo.train(data)
        self.is_ensemble_trained = True
        self.avg_edge_adj = self._get_avg_edge_adj()
        self.est_skeleton_union, self.edge_frequency = self._get_est_skeleton_union()
        self.est_directed_union, self.directed_frequency = (
            self._get_est_directed_union()
        )

    def _get_avg_edge_adj(self):
        if not self.algorithms:
            raise RuntimeError("No algorithms in the ensemble.")

        if not self.is_ensemble_trained:
            raise RuntimeError("Ensemble was not trained yet.")

        first_adj = self.algorithms[0].est_edge_adj
        self.avg_edge_adj = np.zeros(first_adj.shape)
        for algo in self.algorithms:
            self.avg_edge_adj += algo.est_edge_adj
        self.avg_edge_adj /= len(self.algorithms)

        return self.avg_edge_adj

    def _get_est_skeleton_union(self):
        """
        Create a graph that is the union of all algorithm skeletons with edge frequency labels.

        Returns:
            GeneralGraph: A graph containing all edges that appear in at least one algorithm's skeleton
        """
        if not self.algorithms:
            raise RuntimeError("No algorithms in the ensemble.")

        if not self.is_ensemble_trained:
            raise RuntimeError("Ensemble was not trained yet.")

        orig_nodes = self.algorithms[0].est_graph_skeleton.get_nodes()
        new_nodes = [GraphNode(node.get_name()) for node in orig_nodes]
        union_graph = GeneralGraph(new_nodes)
        edge_counts = {}

        for algo in self.algorithms:
            skeleton = algo.est_graph_skeleton
            for i in range(len(orig_nodes)):
                for j in range(i + 1, len(orig_nodes)):
                    if skeleton.is_adjacent_to(orig_nodes[i], orig_nodes[j]):
                        edge_key = (i, j)

                        if edge_key in edge_counts:
                            edge_counts[edge_key] += 1
                        else:
                            edge_counts[edge_key] = 1
                            edge = Edge(
                                new_nodes[i], new_nodes[j], Endpoint.TAIL, Endpoint.TAIL
                            )
                            union_graph.add_edge(edge)

        edge_frequency = self._get_frequency_from_count(
            edge_counts,
            len(self.algorithms),
        )

        return union_graph, edge_frequency

    def _get_est_directed_union(self):
        """
        Create a graph that is the union of all directed edges from all algorithms with arrow frequency labels.

        Returns:
            Tuple[GeneralGraph, Dict]: A graph containing all directed edges that appear in at least
                                      one algorithm's result, and a dictionary with direction frequencies
        """
        if not self.algorithms:
            raise RuntimeError("No algorithms in the ensemble.")

        if not self.is_ensemble_trained:
            raise RuntimeError("Ensemble was not trained yet.")

        orig_nodes = self.algorithms[0].est_graph_skeleton.get_nodes()
        new_nodes = [GraphNode(node.get_name()) for node in orig_nodes]
        directed_union = GeneralGraph(new_nodes)

        arrow_counts = {}

        for algo in self.algorithms:
            graph = algo.est_graph
            for i in range(len(orig_nodes)):
                for j in range(len(orig_nodes)):
                    if i == j:
                        continue
                    if (
                        graph.get_endpoint(
                            orig_nodes[i],
                            orig_nodes[j],
                        )
                        == Endpoint.ARROW
                    ):
                        edge_key = (i, j)
                        edge = Edge(
                            new_nodes[i],
                            new_nodes[j],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                        directed_union.add_edge(edge)

                        if edge_key in arrow_counts:
                            arrow_counts[edge_key] += 1
                        else:
                            arrow_counts[edge_key] = 1

        edge_direction_frequency = self._get_frequency_from_count(
            arrow_counts,
            len(self.algorithms),
        )

        return directed_union, edge_direction_frequency

    def _create_filtered_skeleton_graph(self, threshold: float = 0.5) -> GeneralGraph:
        """
        Create a filtered skeleton graph containing only edges with frequency above threshold.

        Parameters:
        -----------
        threshold : float
            Minimum frequency required to include an edge (between 0 and 1)

        Returns:
        --------
        GeneralGraph
            A graph containing only edges above the threshold frequency
        """
        if not self.is_ensemble_trained:
            raise RuntimeError("Ensemble was not trained yet.")

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        nodes = self.est_skeleton_union.get_nodes()
        filtered_graph = GeneralGraph(nodes)
        filtered_edge_labels = {}

        for (i, j), freq_str in self.edge_frequency.items():
            frequency = float(freq_str)
            if frequency >= threshold:
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                filtered_graph.add_edge(edge)
                filtered_edge_labels[(i, j)] = freq_str

        return filtered_graph, filtered_edge_labels

    def _create_filtered_directed_graph(
        self, threshold: float = 0.5
    ) -> Tuple[GeneralGraph, Dict]:
        """
        Create a filtered directed graph containing only directed edges with frequency above threshold.

        Parameters:
        -----------
        threshold : float
            Minimum frequency required to include a directed edge (between 0 and 1)

        Returns:
        --------
        Tuple[GeneralGraph, Dict]
            A graph containing only directed edges above the threshold frequency,
            and a dictionary with the frequency labels
        """
        if not self.is_ensemble_trained:
            raise RuntimeError("Ensemble was not trained yet.")

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        nodes = self.est_directed_union.get_nodes()
        filtered_graph = GeneralGraph(nodes)
        filtered_edge_labels = {}

        for (i, j), freq_str in self.directed_frequency.items():
            frequency = float(freq_str)
            if frequency >= threshold:
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                filtered_graph.add_edge(edge)
                filtered_edge_labels[(i, j)] = freq_str

        return filtered_graph, filtered_edge_labels

    @staticmethod
    def _get_frequency_from_count(
        count: dict,
        denominator: int,
    ) -> dict:
        frequency = {}
        for (i, j), count in count.items():
            frequency[(i, j)] = round(count / denominator, 3)
        return frequency

    def plot_ensemble_skeleton(
        self,
        type: str = "skeleton",
        threshold: float = 0.5,
        title: str = None,
        fpath: Optional[str] = None,
    ):
        """
        Plot the ensemble skeleton with edges filtered by frequency threshold.

        Parameters:
        -----------
        threshold : float
            Minimum frequency required to include an edge (between 0 and 1)
        title : str, optional
            Title for the plot. If None, a default title will be used.
        fpath : str, optional
            Path to save the figure

        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        if type not in ("skeleton", "directed"):
            raise ValueError(f"{type} is not a valid graph type.")

        if not title:
            title = f"Ensemble {type} (threshold = {threshold})"

        if type == "skeleton":
            filtered_graph, edge_labels = self._create_filtered_skeleton_graph(
                threshold
            )
        elif type == "directed":
            filtered_graph, edge_labels = self._create_filtered_directed_graph(
                threshold
            )

        plotter = Plotter()
        fig = plotter.plot_labeled_graph(
            graph=filtered_graph, edge_labels=edge_labels, title=title, fpath=fpath
        )

        return fig
