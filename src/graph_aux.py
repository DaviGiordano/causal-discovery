from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge
from causallearn.graph.Node import Node
import numpy as np


def dag_adj_to_graph(
    dag_adj: np.ndarray,
    adj_type: str = "upper_triangular",
) -> GeneralGraph:
    nodes = list[Node]([GraphNode(f"X{i}") for i in range(1, dag_adj.shape[0] + 1)])
    graph = GeneralGraph(nodes)

    for i in range(dag_adj.shape[0]):
        for j in range(dag_adj.shape[1]):
            if dag_adj[i, j] != 0 and not np.isnan(dag_adj[i, j]):
                if adj_type == "lower_triangular":
                    graph.add_directed_edge(graph.nodes[j], graph.nodes[i])
                elif adj_type == "upper_triangular":
                    graph.add_directed_edge(graph.nodes[i], graph.nodes[j])

    return graph


def get_graph_skeleton(original_graph: GeneralGraph) -> GeneralGraph:
    """
    Creates and returns the skeleton (undirected graph) from a GeneralGraph.

    Parameters:
    -----------
    original_graph : GeneralGraph
        The original graph

    Returns:
    --------
    GeneralGraph
        The skeleton with only undirected edges
    """
    # Create a new graph with the same nodes
    nodes = original_graph.get_nodes()
    skeleton = GeneralGraph(nodes)

    # For each pair of nodes, add an undirected edge if they're adjacent in the original graph
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if original_graph.is_adjacent_to(nodes[i], nodes[j]):
                # Create an undirected edge directly
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                skeleton.add_edge(edge)

    return skeleton
