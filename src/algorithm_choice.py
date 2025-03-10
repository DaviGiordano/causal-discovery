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


def get_discovery_algorithm(algorithm_name: str, algorithm_params: dict):
    if algorithm_name == "pc":
        return PCAlgorithm(algorithm_params)
    elif algorithm_name == "fci":
        return FCIAlgorithm(algorithm_params)
    elif algorithm_name == "ges":
        return GESAlgorithm(algorithm_params)
    elif algorithm_name == "es":
        return ExactSearchAlgorithm(algorithm_params)
    elif algorithm_name == "icalingam":
        return ICALiNGAMAlgorithm(algorithm_params)
    elif algorithm_name == "directlingam":
        return DirectLiNGAMAlgorithm(algorithm_params)
    elif algorithm_name == "grasp":
        return GRaSPAlgorithm(algorithm_params)
    elif algorithm_name == "boss":
        return BossAlgorithm(algorithm_params)
    elif algorithm_name == "notears":
        return NOTEARSAlgorithm(algorithm_params)
    elif algorithm_name == "dag_gnn":
        return DAGGNNAlgorithm(algorithm_params)
    elif algorithm_name == "corl":
        return CORLAlgorithm(algorithm_params)
    elif algorithm_name == "grandag":
        return GraNDAGAlgorithm(algorithm_params)
    else:
        raise NotImplementedError(f"{algorithm_name} was not yet implemented")
