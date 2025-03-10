from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.search.FCMBased import lingam
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss

from src.causal_discovery.CausalDiscoveryAlgorithm import CausalDiscoveryAlgorithm
from src.graph_aux import dag_adj_to_graph
from src.logging_config import setup_logging
import logging

# setup_logging()
logger = logging.getLogger(__name__)


class PCAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running PC with params: {config_params}")

    def train(self, data) -> None:
        cg = pc(
            data,
            **self.config_params,
        )
        self.est_graph = cg.G
        self.est_adj = None


class FCIAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running FCI with params: {config_params}")

    def train(self, data) -> None:
        G, edges = fci(
            data,
            **self.config_params,
        )
        self.est_graph = G
        self.est_adj = None


class GESAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running GES with params: {config_params}")

    def train(self, data) -> None:
        result = ges(
            data,
            **self.config_params,
        )
        self.est_graph = result["G"]
        self.est_adj = None


class ExactSearchAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running Exact Search with params: {config_params}")

    def train(self, data) -> None:
        dag_adj, _ = bic_exact_search(
            X=data,
            **self.config_params,
        )
        self.est_adj = dag_adj
        self.est_graph = dag_adj_to_graph(dag_adj)


class ICALiNGAMAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running ICALiNGAM with params: {config_params}")

    def train(self, data) -> None:
        model = lingam.ICALiNGAM(
            **self.config_params,
        )
        model.fit(data)
        self.est_adj = model.adjacency_matrix_
        self.est_graph = dag_adj_to_graph(self.est_adj)


class DirectLiNGAMAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running DirectLiNGAM with params: {config_params}")

    def train(self, data) -> None:
        model = lingam.DirectLiNGAM(
            **self.config_params,
        )
        model.fit(data)
        self.est_adj = model.adjacency_matrix_
        self.est_graph = dag_adj_to_graph(self.est_adj, "lower_triangular")


class GRaSPAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running GRaSP with params: {config_params}")

    def train(self, data) -> None:
        G = grasp(
            data,
            **self.config_params,
        )
        self.est_graph = G
        self.est_adj = None


class BossAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running BOSS with params: {config_params}")

    def train(self, data) -> None:
        G = boss(
            data,
            **self.config_params,
        )
        self.est_graph = G
        self.est_adj = None
