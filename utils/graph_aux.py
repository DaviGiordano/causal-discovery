from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
import numpy as np


def dag_adj_to_graph(dag_adj):
    nodes = [GraphNode(f"X{i}") for i in range(1, dag_adj.shape[0] + 1)]
    graph = GeneralGraph(nodes)

    for i in range(dag_adj.shape[0]):
        for j in range(dag_adj.shape[1]):
            if dag_adj[i, j] != 0 and not np.isnan(dag_adj[i, j]):
                graph.add_directed_edge(graph.nodes[i], graph.nodes[j])

    return graph
