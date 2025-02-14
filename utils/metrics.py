from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion


def get_graph_confusion(graph_type, true_g, est_g):
    """Calculate confusion matrix, precision and recall for different graph representations.

    Args:
        graph_type (str): Type of graph comparison ('adj', 'arrow', or 'arrow_ce')
        true_g (GeneralGraph): True graph
        est_g (GeneralGraph): Estimated graph

    Returns:
        tuple: (confusion_matrix, precision, recall)
    """
    # Validate inputs
    if not true_g or not est_g:
        raise ValueError("Both true and estimated graphs must be provided")

    if not true_g.get_nodes() or not est_g.get_nodes():
        raise ValueError("Both graphs must have nodes")

    # Ensure both graphs have the same nodes
    true_nodes = set(node.get_name() for node in true_g.get_nodes())
    est_nodes = set(node.get_name() for node in est_g.get_nodes())

    if true_nodes != est_nodes:
        raise ValueError(
            f"Graphs have different nodes.\nTrue: {true_nodes}\nEstimated: {est_nodes}"
        )

    if graph_type == "arrow":
        arrow = ArrowConfusion(true_g, est_g)
        tp = arrow.get_arrows_tp()
        fp = arrow.get_arrows_fp()
        fn = arrow.get_arrows_fn()
        tn = arrow.get_arrows_tn()
        precision = round(arrow.get_arrows_precision(), 2)
        recall = round(arrow.get_arrows_recall(), 2)

        try:
            precision = round(arrow.get_arrows_precision_ce(), 2)
        except ZeroDivisionError:
            precision = "err"
        try:
            recall = round(arrow.get_arrows_recall_ce(), 2)
        except ZeroDivisionError:
            recall = "err"

    elif graph_type == "arrow_ce":
        arrow = ArrowConfusion(true_g, est_g)
        tp = arrow.get_arrows_tp_ce()
        fp = arrow.get_arrows_fp_ce()
        fn = arrow.get_arrows_fn_ce()
        tn = arrow.get_arrows_tn_ce()

        try:
            precision = round(arrow.get_arrows_precision_ce(), 2)
        except ZeroDivisionError:
            precision = "err"
        try:
            recall = round(arrow.get_arrows_recall_ce(), 2)
        except ZeroDivisionError:
            recall = "err"

    elif graph_type == "adj":
        adj = AdjacencyConfusion(true_g, est_g)
        tp = adj.get_adj_tp()
        fp = adj.get_adj_fp()
        fn = adj.get_adj_fn()
        tn = adj.get_adj_tn()

        try:
            precision = round(adj.get_adj_precision(), 2)
        except ZeroDivisionError:
            precision = -1
        try:
            recall = round(adj.get_adj_recall(), 2)
        except ZeroDivisionError:
            recall = -1

    cm = [
        [int(tp), int(fp)],
        [int(fn), int(tn)],
    ]

    return cm, precision, recall
