from typing import Sequence
from causallearn.graph.Node import Node
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge
from causallearn.graph.Node import Node
import numpy as np


def dag_adj_to_graph(
    dag_adj: np.ndarray,
    adj_type: str = "lower_triangular",
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
