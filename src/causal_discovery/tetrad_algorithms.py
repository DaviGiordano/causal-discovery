import json
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

SCORE_AND_TEST_METHODS = {
    "fisherz": {
        "func_name": "use_fisher_z",
        "default_params": {
            "alpha": 0.01,
            "use_for_mc": False,
            "singularity_lambda": 0.0,
        },
    },
    "conditional_gaussian_test": {
        "func_name": "use_conditional_gaussian_test",
        "default_params": {
            "alpha": 0.01,
            "discretize": True,
            "use_for_mc": False,
        },
    },
    "conditional_gaussian_score": {
        "func_name": "use_conditional_gaussian_score",
        "default_params": {
            "penalty_discount": 1,
            "discretize": True,
            "num_categories_to_discretize": 3,
            "structure_prior": 0,
        },
    },
}


class TetradAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)

    def _set_test_or_score(self, search):
        test_or_score_name = self.config_params.get("test_or_score_name", "")
        test_or_score_params = self.config_params.get("test_or_score_params", {})
        func_name = SCORE_AND_TEST_METHODS[test_or_score_name]["func_name"]
        default_params = SCORE_AND_TEST_METHODS[test_or_score_name]["default_params"]

        for key, value in default_params.items():
            if key not in test_or_score_params:
                test_or_score_params[key] = value
        logger.info(
            f"Final test or score params:\n{json.dumps(test_or_score_params, indent=2)}"
        )
        try:
            method = getattr(search, func_name)
            method(**test_or_score_params)
        except Exception as e:
            raise ValueError(
                f"Error in configuration of test or score: '{test_or_score_name}'. "
                f"Config params: {self.config_params}"
                f"{e}"
            )

    def _set_bootstrap(self, search):
        bootstrap_params = self.config_params.get("bootstrap_params", {})

        number_resampling = bootstrap_params.get("numberResampling", None)
        percent_resample = bootstrap_params.get("percent_resample_size", None)
        add_original = bootstrap_params.get("add_original", None)
        with_replacement = bootstrap_params.get("with_replacement", None)
        resampling_ensemble = bootstrap_params.get("resampling_ensemble", None)
        seed = bootstrap_params.get("seed", None)

        if any(
            param is None
            for param in [
                number_resampling,
                percent_resample,
                add_original,
                with_replacement,
                resampling_ensemble,
                seed,
            ]
        ):
            error_msg = "Missing required bootstrap parameters"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Set each parameter explicitly
        final_bootstrap_params = {
            "numberResampling": number_resampling,
            "percent_resample_size": percent_resample,
            "add_original": add_original,
            "with_replacement": with_replacement,
            "resampling_ensemble": resampling_ensemble,
            "seed": seed,
        }

        try:
            logger.info(
                f"Configuring bootstrap with params:\n{json.dumps(final_bootstrap_params, indent=2)}"
            )
            search.set_bootstrapping(**final_bootstrap_params)
        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        raise NotImplementedError()

    def train(self, data: np.ndarray, node_names: list = []) -> None:

        if not node_names:
            node_names = [f"X{i+1}" for i in range(data.shape[1])]

        df = pd.DataFrame(data, columns=node_names)

        search = TetradSearch(df)
        search.set_verbose(False)

        self._set_test_or_score(search)

        if self.config_params.get("bootstrap_params"):
            self._set_bootstrap(search)

        self._algorithm_specific_train(search)

        self.graph_string = str(search.get_string())
        self.est_dotgraph = search.get_dot()
        self.est_graph = str_to_general_graph(self.graph_string)
        self.edge_probabilities = str_to_edge_probabilities(self.graph_string)
        self.est_edges_dict = str_to_edge_dict(self.graph_string)
        self._set_auxiliary_results()


class PCTetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(
            f"Instantiated algorithm with params:\n{json.dumps(self.config_params, indent=2)}"
        )

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        conflict_rule = self.config_params.get("algorithm_params", {}).get(
            "conflict_rule", 1
        )
        depth = self.config_params.get("algorithm_params", {}).get("depth", -1)
        stable_fas = self.config_params.get("algorithm_params", {}).get(
            "stable_fas", True
        )
        guarantee_cpdag = self.config_params.get("algorithm_params", {}).get(
            "guarantee_cpdag", False
        )
        params_dict = {
            "conflict_rule": conflict_rule,
            "depth": depth,
            "stable_fas": stable_fas,
            "guarantee_cpdag": guarantee_cpdag,
        }
        try:
            logger.info(f"Running PC with params:\n{json.dumps(params_dict, indent=2)}")
            search.run_pc(**params_dict)

        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search


class BOSSTetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(f"Instantiated BOSS algorithm with params:\n{self.config_params}")

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        try:
            # Extract BOSS-specific parameters with defaults
            algorithm_params = self.config_params.get("algorithm_params", {})

            num_starts = algorithm_params.get("num_starts", 1)
            use_bes = algorithm_params.get("use_bes", False)
            time_lag = algorithm_params.get("time_lag", 0)
            use_data_order = algorithm_params.get("use_data_order", True)
            output_cpdag = algorithm_params.get("output_cpdag", True)

            logger.info(
                f"Running BOSS with params: num_starts={num_starts}, use_bes={use_bes}, "
                f"time_lag={time_lag}, use_data_order={use_data_order}, output_cpdag={output_cpdag}"
            )

            # Run BOSS algorithm with extracted parameters
            search.run_boss(
                num_starts=num_starts,
                use_bes=use_bes,
                time_lag=time_lag,
                use_data_order=use_data_order,
                output_cpdag=output_cpdag,
            )
        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search
