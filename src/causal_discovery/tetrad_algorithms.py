import jpype
from src.graph_aux import dag_adj_to_graph
from src.causal_discovery.CausalDiscoveryAlgorithm import CausalDiscoveryAlgorithm
from src.pytetrad.TetradSearch import TetradSearch
import pandas as pd
import numpy as np
import logging

from src.parse_tetrad_string import (
    str_to_edge_dict,
    str_to_edge_probabilities,
    str_to_general_graph,
)

logger = logging.getLogger(__name__)

INDEP_TEST_METHODS = {
    "fisherz": "use_fisher_z",
    "conditional_gaussian": "use_conditional_gaussian_test",
    "degenerate_gaussian": "use_degenerate_gaussian_score",
}


class PCTetrad(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)

    def _set_indep_test(self, search):
        algorithm_params = self.config_params.get("algorithm_params", {})
        indep_test = algorithm_params.pop("indep_test", None)

        try:
            method_name = INDEP_TEST_METHODS[indep_test]
            method = getattr(search, method_name)
            method(**algorithm_params)

        except KeyError:
            raise NotImplementedError(
                f"Independence test '{indep_test}' not found. "
                f"Config params: {self.config_params}"
            )

    def _set_bootstrap(self, search):
        bootstrap_params = self.config_params.get("bootstrap_params", {})
        search.set_bootstrapping(**bootstrap_params)

    def train(self, data: np.ndarray, node_names: list = []) -> None:
        logger.info(f"Running PC in tetrad with params: {self.config_params}")

        if not node_names:
            node_names = [f"X{i+1}" for i in range(data.shape[1])]

        df = pd.DataFrame(data, columns=node_names)
        try:

            search = TetradSearch(df)
            search.set_verbose(False)

            self._set_indep_test(search)
            if self.config_params.get("bootstrap_params"):
                self._set_bootstrap(search)

            search.run_pc()
        except jpype.JException as ex:
            print("Java exception:", ex.message())
            raise

        self.graph_string = str(search.get_string())
        self.est_dotgraph = search.get_dot()
        self.est_graph = str_to_general_graph(self.graph_string)
        self.edge_probabilities = str_to_edge_probabilities(self.graph_string)
        self.est_edges_dict = str_to_edge_dict(self.graph_string)
        self._set_auxiliary_results()
