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
