from castle.algorithms import Notears, DAG_GNN, CORL, GraNDAG
from src.graph_aux import dag_adj_to_graph
from src.causal_discovery.CausalDiscoveryAlgorithm import CausalDiscoveryAlgorithm
from src.logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


class NOTEARSAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running NOTEARS with params: {config_params}")

    def train(self, data) -> None:
        model = Notears()
        model.learn(data)
        self.est_adj = model.causal_matrix
        self.est_graph = dag_adj_to_graph(model.causal_matrix)


class DAGGNNAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running DAG-GNN with params: {config_params}")

    def train(self, data) -> None:
        model = DAG_GNN(**self.config_params)
        model.learn(data)
        self.est_adj = model.causal_matrix
        self.est_graph = dag_adj_to_graph(model.causal_matrix)


class CORLAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running CORL with params: {config_params}")

    def train(self, data) -> None:
        model = CORL(**self.config_params)
        model.learn(data)
        self.est_adj = model.causal_matrix
        self.est_graph = dag_adj_to_graph(model.causal_matrix)


class GraNDAGAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(msg=f"Running GraNDAG with params: {config_params}")

    def train(self, data) -> None:
        model = GraNDAG(input_dim=data.shape[1], **self.config_params)
        model.learn(data)
        self.est_adj = model.causal_matrix
        self.est_graph = dag_adj_to_graph(model.causal_matrix)
