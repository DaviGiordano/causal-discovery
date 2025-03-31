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


def get_discovery_algorithm(**algorithm_config) -> CausalDiscoveryAlgorithm:
    algorithm_name = algorithm_config.pop("algorithm_name", "")

    if algorithm_name == "pc":
        return PCAlgorithm(algorithm_config)
    elif algorithm_name == "fci":
        return FCIAlgorithm(algorithm_config)
    elif algorithm_name == "ges":
        return GESAlgorithm(algorithm_config)
    elif algorithm_name == "es":
        return ExactSearchAlgorithm(algorithm_config)
    elif algorithm_name == "icalingam":
        return ICALiNGAMAlgorithm(algorithm_config)
    elif algorithm_name == "directlingam":
        return DirectLiNGAMAlgorithm(algorithm_config)
    elif algorithm_name == "grasp":
        return GRaSPAlgorithm(algorithm_config)
    elif algorithm_name == "boss":
        return BossAlgorithm(algorithm_config)
    elif algorithm_name == "notears":
        return NOTEARSAlgorithm(algorithm_config)
    elif algorithm_name == "dag_gnn":
        return DAGGNNAlgorithm(algorithm_config)
    elif algorithm_name == "corl":
        return CORLAlgorithm(algorithm_config)
    elif algorithm_name == "grandag":
        return GraNDAGAlgorithm(algorithm_config)
    elif algorithm_name == "pc_tetrad":
        return PCTetrad(algorithm_config)
    else:
        raise NotImplementedError(f"{algorithm_name} was not yet implemented")
