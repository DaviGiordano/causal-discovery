from src.causal_discovery.CausalDiscoveryAlgorithm import CausalDiscoveryAlgorithm
from src.causal_discovery.causallearn_algorithms import (
    PCAlgorithm,
    FCIAlgorithm,
    GESAlgorithm,
    ExactSearchAlgorithm,
    ICALiNGAMAlgorithm,
    DirectLiNGAMAlgorithm,
    GRaSPAlgorithm,
    BossAlgorithm,
)

from src.causal_discovery.castle_algorithms import (
    NOTEARSAlgorithm,
    DAGGNNAlgorithm,
    CORLAlgorithm,
    GraNDAGAlgorithm,
)

from src.causal_discovery.tetrad_algorithms import PCTetrad


def get_discovery_algorithm(**config_params) -> CausalDiscoveryAlgorithm:
    algorithm_name = config_params.pop("algorithm_name", "")

    if algorithm_name == "pc":
        return PCAlgorithm(config_params)
    elif algorithm_name == "fci":
        return FCIAlgorithm(config_params)
    elif algorithm_name == "ges":
        return GESAlgorithm(config_params)
    elif algorithm_name == "es":
        return ExactSearchAlgorithm(config_params)
    elif algorithm_name == "icalingam":
        return ICALiNGAMAlgorithm(config_params)
    elif algorithm_name == "directlingam":
        return DirectLiNGAMAlgorithm(config_params)
    elif algorithm_name == "grasp":
        return GRaSPAlgorithm(config_params)
    elif algorithm_name == "boss":
        return BossAlgorithm(config_params)
    elif algorithm_name == "notears":
        return NOTEARSAlgorithm(config_params)
    elif algorithm_name == "dag_gnn":
        return DAGGNNAlgorithm(config_params)
    elif algorithm_name == "corl":
        return CORLAlgorithm(config_params)
    elif algorithm_name == "grandag":
        return GraNDAGAlgorithm(config_params)
    elif algorithm_name == "pc_tetrad":
        return PCTetrad(config_params)
    else:
        raise NotImplementedError(f"{algorithm_name} was not yet implemented")
