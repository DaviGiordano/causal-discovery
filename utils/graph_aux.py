from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
import numpy as np


def convert_adj_matrix_to_tetrad_file(
    adj_matrix: np.ndarray,
    output_fpath: str = None,
) -> None:

    n = adj_matrix.shape[0]
    node_names = [f"X{i+1}" for i in range(n)]

    with open(output_fpath, "w") as f:
        # Write the header line so that words[1] will be "Nodes:"
        f.write("Graph Nodes:\n")
        # Write the nodes line: list nodes separated by semicolons.
        f.write(";".join(node_names) + "\n")
        # Write each edge line.
        # For every nonzero entry in the matrix, output an edge.
        # We assume a 1 at position (i, j) means an edge from node_names[i] to node_names[j].
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] == 1:
                    # The edge line follows the pattern: "Edge. X_i -> X_j"
                    f.write(f"Edge. {node_names[i]} -> {node_names[j]}\n")
    print(f"Tetrad file written to {output_fpath}")


def dag_adj_to_graph(dag_adj):
    # Create node objects
    nodes = [GraphNode(f"X{i}") for i in range(1, dag_adj.shape[0] + 1)]

    # Create a GeneralGraph object
    graph = GeneralGraph(nodes)

    # Add edges based on adjacency matrix
    for i in range(dag_adj.shape[0]):
        for j in range(dag_adj.shape[1]):
            if dag_adj[i, j] != 0 and not np.isnan(dag_adj[i, j]):
                # Add directed edge from node i to node j
                graph.add_directed_edge(graph.nodes[i], graph.nodes[j])

    return graph
