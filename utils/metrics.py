from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion


def get_graph_confusion(type, true_g, est_g):
    """
    Calculate the confusion matrix, precision, and recall for a given graph type.

    Parameters:
    type (str): The type of graph confusion to calculate. Can be "arrow" or "adj".
    true_g (Graph): The ground truth graph.
    est_g (Graph): The estimated graph.

    Returns:
    tuple: A tuple containing:
        - cm (list of list of int): The confusion matrix in the format [[TP, FP], [FN, TN]].
        - precision (float): The precision of the estimated graph.
        - recall (float): The recall of the estimated graph.
    """
    if type == "arrow":
        arrow = ArrowConfusion(true_g, est_g)
        tp = arrow.get_arrows_tp()
        fp = arrow.get_arrows_fp()
        fn = arrow.get_arrows_fn()
        tn = arrow.get_arrows_tn()
        precision = round(arrow.get_arrows_precision(), 2)
        recall = round(arrow.get_arrows_recall(), 2)
    elif type == "arrow_ce":
        arrow = ArrowConfusion(true_g, est_g)
        tp = arrow.get_arrows_tp_ce()
        fp = arrow.get_arrows_fp_ce()
        fn = arrow.get_arrows_fn_ce()
        tn = arrow.get_arrows_tn_ce()
        precision = round(arrow.get_arrows_precision_ce(), 2)
        recall = round(arrow.get_arrows_recall_ce(), 2)

    elif type == "adj":
        adj = AdjacencyConfusion(true_g, est_g)
        tp = adj.get_adj_tp()
        fp = adj.get_adj_fp()
        fn = adj.get_adj_fn()
        tn = adj.get_adj_tn()
        precision = round(adj.get_adj_precision(), 2)
        recall = round(adj.get_adj_recall(), 2)

    cm = [
        [int(tp), int(fp)],
        [int(fn), int(tn)],
    ]

    return cm, precision, recall
