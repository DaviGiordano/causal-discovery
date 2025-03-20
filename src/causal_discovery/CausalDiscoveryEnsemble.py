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

    def train(self, data: np.ndarray):
        for algo in self.algorithms:
            algo.train(data)
        self.is_ensemble_trained = True
        self.avg_edge_adj = self._get_avg_edge_adj()
        self.est_skeleton_union = self._get_est_skeleton_union()

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
                            # Use the new nodes when adding edges
                            edge = Edge(
                                new_nodes[i], new_nodes[j], Endpoint.TAIL, Endpoint.TAIL
                            )
                            union_graph.add_edge(edge)

        # Create edge labels showing frequency
        self.edge_frequency = {}
        for (i, j), count in edge_counts.items():
            frequency = count / len(self.algorithms)
            self.edge_frequency[(i, j)] = f"{frequency:.2f}"

        # Save and return the union graph
        self.est_graph_union = union_graph
        return union_graph

    def _get_filtered_skeleton_labels(self, threshold: float = 0.5) -> GeneralGraph:
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

        # Create a new graph with the same nodes
        nodes = self.est_graph_union.get_nodes()
        filtered_graph = GeneralGraph(nodes)
        edge_labels = {}

        # Add only edges with frequency above threshold
        for (i, j), freq_str in self.edge_frequency.items():
            frequency = float(freq_str)
            if frequency >= threshold:
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                filtered_graph.add_edge(edge)
                edge_labels[(i, j)] = freq_str

        return filtered_graph, edge_labels

    def plot_ensemble_skeleton(
        self, threshold: float = 0.5, title: str = None, fpath: Optional[str] = None
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
        if not title:
            title = f"Ensemble Skeleton (threshold = {threshold})"

        # Get filtered skeleton graph
        filtered_graph, edge_labels = self._get_filtered_skeleton_labels(threshold)

        # Create edge labels dictionary for edges that are in the filtered graph

        plotter = Plotter()

        fig = plotter.plot_labeled_graph(
            graph=filtered_graph, edge_labels=edge_labels, title=title, fpath=fpath
        )

        return fig
