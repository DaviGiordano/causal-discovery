from causallearn.utils.GraphUtils import GraphUtils
from utils.metrics import get_graph_confusion
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import seaborn as sns
import json


def print_dict(dictionary):
    print(json.dumps(dictionary, indent=4))


def plot_confusion(
    cm,
    title="Confusion Matrix",
    xlabel="Predicted Label",
    ylabel="Actual Label",
    xticklabels=["Predicted Positive", "Predicted Negative"],
    yticklabels=["Actual Positive", "Actual Negative"],
    ax=None,
):
    """
    Plots a confusion matrix from a 2x2 confusion matrix.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
    else:
        fig = ax.figure

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_confusion_comparison(true_g, est_g, title="Edge and Arrow CMs", fpath=None):
    cm_adj, prec_adj, rec_adj = get_graph_confusion("adj", true_g, est_g)
    cm_arrow, prec_arrow, rec_arrow = get_graph_confusion("arrow", true_g, est_g)
    cm_arrow_ce, prec_arrow_ce, rec_arrow_ce = get_graph_confusion(
        "arrow_ce", true_g, est_g
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    plot_confusion(
        cm_adj,
        title=f"Edge CM | precision={prec_adj} | recall={rec_adj}",
        xlabel="Predicted Edges",
        ylabel="Actual Edges",
        xticklabels=["", ""],
        yticklabels=["", ""],
        ax=ax1,
    )
    plot_confusion(
        cm_arrow,
        title=f"Arrow CM | precision={prec_arrow} | recall={rec_arrow}",
        xlabel="Predicted Arrows",
        ylabel="Actual Arrows",
        xticklabels=["", ""],
        yticklabels=["", ""],
        ax=ax2,
    )

    plot_confusion(
        cm_arrow_ce,
        title=f"Arrow_ce CM | precision={prec_arrow_ce} | recall={rec_arrow_ce}",
        xlabel="Predicted Arrows",
        ylabel="Actual Arrows",
        xticklabels=["", ""],
        yticklabels=["", ""],
        ax=ax3,
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if fpath:
        fig.savefig(fpath)

    return fig


def plot_graph_comparison(true_g, est_g, fpath=None):
    true_pyd = GraphUtils.to_pydot(true_g)
    true_pyd.set_rankdir("LR")
    true_g_img = Image.open(BytesIO(true_pyd.create_png()))

    estimated_pyd = GraphUtils.to_pydot(est_g)
    estimated_pyd.set_rankdir("LR")
    est_g_img = Image.open(BytesIO(estimated_pyd.create_png()))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.05)
    divider = fig.add_axes([0.5, 0.1, 0.01, 0.8])
    divider.axvline(0.5, color="black", linewidth=1)
    divider.axis("off")

    ax_left.imshow(true_g_img)
    ax_left.axis("off")
    ax_left.set_title("True Graph")

    ax_right.imshow(est_g_img)
    ax_right.axis("off")
    ax_right.set_title("Estimated Graph")

    if fpath:
        fig.savefig(fpath)

    return fig
