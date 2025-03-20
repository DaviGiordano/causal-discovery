from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge
from causallearn.graph.Node import Node

from typing import List, Optional, Dict, Tuple
import pydot
import numpy as np


def dag_adj_to_graph(
    dag_adj: np.ndarray,
    adj_type: str = "upper_triangular",
) -> GeneralGraph:
    nodes = list[Node]([GraphNode(f"X{i}") for i in range(1, dag_adj.shape[0] + 1)])
    graph = GeneralGraph(nodes)

    if adj_type == "lower_triangular":
        for i in range(dag_adj.shape[0]):
            for j in range(dag_adj.shape[1]):
                if dag_adj[i, j] != 0 and not np.isnan(dag_adj[i, j]):
                    graph.add_directed_edge(graph.nodes[j], graph.nodes[i])
    elif adj_type == "upper_triangular":
        for i in range(dag_adj.shape[0]):
            for j in range(dag_adj.shape[1]):
                if dag_adj[i, j] != 0 and not np.isnan(dag_adj[i, j]):
                    graph.add_directed_edge(graph.nodes[i], graph.nodes[j])

    return graph


def get_edge_adjacency_matrix(graph: GeneralGraph) -> np.ndarray:
    """
    Creates a symmetrical adjacency matrix from a graph, ignoring edge directions.

    Parameters:
    -----------
    graph : GeneralGraph
        The input graph

    Returns:
    --------
    np.ndarray
        A symmetrical adjacency matrix where 1 indicates an edge between nodes
    """
    nodes = graph.get_nodes()
    n = len(nodes)
    edge_adj_matrix = np.zeros((n, n), dtype=int)

    # For each pair of nodes, set matrix values to 1 if they're adjacent
    for i in range(n):
        for j in range(n):
            if graph.is_adjacent_to(nodes[i], nodes[j]):
                edge_adj_matrix[i, j] = 1
                edge_adj_matrix[j, i] = 1  # Make it symmetrical

    return edge_adj_matrix


def get_graph_skeleton(graph: GeneralGraph) -> GeneralGraph:
    """
    Creates and returns the skeleton (undirected graph) from a GeneralGraph.

    Parameters:
    -----------
    graph : GeneralGraph
        The original graph

    Returns:
    --------
    GeneralGraph
        The skeleton with only undirected edges
    """
    # Create a new graph with the same nodes
    nodes = graph.get_nodes()
    skeleton = GeneralGraph(nodes)

    # For each pair of nodes, add an undirected edge if they're adjacent in the original graph
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if graph.is_adjacent_to(nodes[i], nodes[j]):
                # Create an undirected edge directly
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                skeleton.add_edge(edge)

    return skeleton
