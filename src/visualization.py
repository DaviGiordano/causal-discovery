import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
from causallearn.utils.GraphUtils import GraphUtils
from typing import Optional
from src.metrics import Metrics
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class Plotter:
    def __init__(self, metrics: Metrics | None):
        """Initialize plotter with metrics object.

        Args:
            metrics: Metrics object containing evaluation results
        """
        self.metrics = metrics
        if metrics is not None:
            self.true_graph = metrics.true_graph
            self.est_graph = metrics.est_graph

    def plot_confusion(
        self,
        cm: list,
        title: str = "Confusion Matrix",
        xlabel: str = "Predicted Label",
        ylabel: str = "Actual Label",
        xticklabels: list = ["Predicted Positive", "Predicted Negative"],
        yticklabels: list = ["Actual Positive", "Actual Negative"],
        ax: Optional[Axes] = None,
    ):
        """Plot a single confusion matrix.

        Args:
            cm: 2x2 confusion matrix
            title: Title for the plot
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            xticklabels: Labels for x-axis ticks
            yticklabels: Labels for y-axis ticks
            ax: Optional matplotlib axes to plot on

        Returns:
            Figure and axes objects
        """
        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = plt.gca()
        else:
            fig = ax.figure

        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return fig, ax

    def plot_confusion_comparison(
        self,
        title: str = "Edge and Arrow Confusion Matrices",
        fpath: Optional[str] = None,
    ) -> Figure:
        """Plot comparison of confusion matrices for adjacency and arrows.

        Args:
            title: Title for the overall figure
            fpath: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        metrics_data = self.metrics.get_result_metrics()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Plot adjacency confusion matrix
        adj_metrics = metrics_data["adjacency"]
        self.plot_confusion(
            adj_metrics["confusion_matrix"],
            title=f"Edge CM | precision={adj_metrics['precision']} | recall={adj_metrics['recall']}",
            xlabel="Predicted Edges",
            ylabel="Actual Edges",
            xticklabels=["", ""],
            yticklabels=["", ""],
            ax=ax1,
        )

        # Plot arrow confusion matrix
        arrow_metrics = metrics_data["arrow"]
        self.plot_confusion(
            arrow_metrics["confusion_matrix"],
            title=f"Arrow CM | precision={arrow_metrics['precision']} | recall={arrow_metrics['recall']}",
            xlabel="Predicted Arrows",
            ylabel="Actual Arrows",
            xticklabels=["", ""],
            yticklabels=["", ""],
            ax=ax2,
        )

        # Plot arrow_ce confusion matrix
        arrow_ce_metrics = metrics_data["arrow_ce"]
        self.plot_confusion(
            arrow_ce_metrics["confusion_matrix"],
            title=f"Arrow_ce CM | precision={arrow_ce_metrics['precision']} | recall={arrow_ce_metrics['recall']}",
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

    def plot_graph_comparison(self, fpath: Optional[str] = None) -> Figure:
        """Plot comparison between true and estimated graphs.

        Args:
            fpath: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Convert true graph to image
        true_pyd = GraphUtils.to_pydot(self.true_graph)
        true_pyd.set_rankdir("LR")
        true_g_img = Image.open(BytesIO(true_pyd.create_png()))

        # Convert estimated graph to image
        estimated_pyd = GraphUtils.to_pydot(self.est_graph)
        estimated_pyd.set_rankdir("LR")
        est_g_img = Image.open(BytesIO(estimated_pyd.create_png()))

        # Create figure with two subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(wspace=0.05)

        # Add divider
        divider = fig.add_axes((0.5, 0.1, 0.01, 0.8))
        divider.axvline(0.5, color="black", linewidth=1)
        divider.axis("off")

        # Plot true graph
        ax_left.imshow(true_g_img)
        ax_left.axis("off")
        ax_left.set_title("True Graph")

        # Plot estimated graph
        ax_right.imshow(est_g_img)
        ax_right.axis("off")
        ax_right.set_title("Estimated Graph")

        if fpath:
            fig.savefig(fpath)

        return fig
