from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.search.FCMBased import lingam
from causallearn.search.PermutationBased.GRaSP import grasp

from src.causal_discovery.CausalDiscoveryAlgorithm import CausalDiscoveryAlgorithm
from src.graph_aux import dag_adj_to_graph
from src.logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


class PCAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running PC with params: {config_params}")

    def train(self, data) -> None:
        cg = pc(
            data,
            alpha=self.config_params["alpha"],
            indep_test=self.config_params["indep_test"],
            stable=self.config_params["stable"],
            uc_rule=self.config_params["uc_rule"],
            uc_priority=self.config_params["uc_priority"],
            mvpc=self.config_params["mvpc"],
            correction_name=self.config_params["correction_name"],
            background_knowledge=self.config_params["background_knowledge"],
            verbose=True,
            show_progress=True,
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
            independence_test_method=self.config_params["independence_test_method"],
            alpha=self.config_params["alpha"],
            depth=self.config_params["depth"],
            max_path_length=self.config_params["max_path_length"],
            background_knowledge=self.config_params["background_knowledge"],
            verbose=True,
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
            score_func=self.config_params["score_func"],
            maxP=self.config_params["maxP"],
            parameters=self.config_params["parameters"],
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
            super_graph=self.config_params["super_graph"],
            search_method=self.config_params["search_method"],
            use_path_extension=self.config_params["use_path_extension"],
            use_k_cycle_heuristic=self.config_params["use_k_cycle_heuristic"],
            k=self.config_params["k"],
            verbose=True,
            max_parents=self.config_params["max_parents"],
        )
        self.est_adj = dag_adj
        self.est_graph = dag_adj_to_graph(dag_adj)


class ICALiNGAMAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running ICALiNGAM with params: {config_params}")

    def train(self, data) -> None:
        model = lingam.ICALiNGAM(
            self.config_params["random_state"], self.config_params["max_iter"]
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
            self.config_params["random_state"],
            self.config_params["prior_knowledge"],
            self.config_params["apply_prior_knowledge_softly"],
            self.config_params["measure"],
        )
        model.fit(data)
        self.est_adj = model.adjacency_matrix_
        self.est_graph = dag_adj_to_graph(self.est_adj)


class GRaSPAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running GRaSP with params: {config_params}")

    def train(self, data) -> None:
        G = grasp(
            data,
            self.config_params["score_func"],
            self.config_params["depth"],
            self.config_params["parameters"],
        )
        self.est_graph = G
        self.est_adj = None
