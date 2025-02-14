from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam

from utils.logging import redirect_stdout_to_logger
from utils.graph_aux import dag_adj_to_graph
from utils.validation import (
    GRASP_REQUIRED_PARAMS,
    ICALINGAM_REQUIRED_PARAMS,
    PC_REQUIRED_PARAMS,
    FCI_REQUIRED_PARAMS,
    GES_REQUIRED_PARAMS,
    ES_REQUIRED_PARAMS,
    DIRECTLINGAM_REQUIRED_PARAMS,
    RCD_REQUIRED_PARAMS,
    validate_params,
)


@validate_params(PC_REQUIRED_PARAMS)
def run_pc(data, config):

    with redirect_stdout_to_logger():
        cg = pc(
            data,
            alpha=config["alpha"],
            indep_test=config["indep_test"],
            stable=config["stable"],
            uc_rule=config["uc_rule"],
            uc_priority=config["uc_priority"],
            mvpc=config["mvpc"],
            correction_name=config["correction_name"],
            background_knowledge=config["background_knowledge"],
            verbose=True,
            show_progress=True,
        )

    return cg.G


@validate_params(FCI_REQUIRED_PARAMS)
def run_fci(data, config):

    with redirect_stdout_to_logger():
        G, edges = fci(
            data,
            independence_test_method=config["independence_test_method"],
            alpha=config["alpha"],
            depth=config["depth"],
            max_path_length=config["max_path_length"],
            background_knowledge=config["background_knowledge"],
            verbose=True,
        )
    return G


@validate_params(GES_REQUIRED_PARAMS)
def run_ges(data, config):

    with redirect_stdout_to_logger():
        G = ges(
            data,
            score_func=config["score_func"],
            maxP=config["maxP"],
            parameters=config["parameters"],
        )["G"]

    return G


@validate_params(ES_REQUIRED_PARAMS)
def run_es(data, config):

    with redirect_stdout_to_logger():
        dag_adj, search_stats = bic_exact_search(
            X=data,
            super_graph=config["super_graph"],
            search_method=config["search_method"],
            use_path_extension=config["use_path_extension"],
            use_k_cycle_heuristic=config["use_k_cycle_heuristic"],
            k=config["k"],
            verbose=True,
            max_parents=config["max_parents"],
        )
        print(f"ES search stats:\n{search_stats}")
        print(f"Adjacency matrix:\n{dag_adj}")

    graph = dag_adj_to_graph(dag_adj)

    return graph


@validate_params(ICALINGAM_REQUIRED_PARAMS)
def run_icalingam(data, config):

    with redirect_stdout_to_logger():
        model = lingam.ICALiNGAM(config["random_state"], config["max_iter"])
        model.fit(data)
        print(f"Causal order:\n{model.causal_order_}")
        print(f"Adjacency matrix:\n{model.adjacency_matrix_}")

    graph = dag_adj_to_graph(model.adjacency_matrix_)

    return graph


@validate_params(DIRECTLINGAM_REQUIRED_PARAMS)
def run_directlingam(data, config):

    with redirect_stdout_to_logger():
        model = lingam.DirectLiNGAM(
            config["random_state"],
            config["prior_knowledge"],
            config["apply_prior_knowledge_softly"],
            config["measure"],
        )
        model.fit(data)
        print(f"Causal order:\n{model.causal_order_}")
        print(f"Adjacency matrix:\n{model.adjacency_matrix_}")

    graph = dag_adj_to_graph(model.adjacency_matrix_)

    return graph


@validate_params(GRASP_REQUIRED_PARAMS)
def run_grasp(data, config):

    with redirect_stdout_to_logger():
        G = grasp(
            data,
            config["score_func"],
            config["depth"],
            config["parameters"],
        )
    return G


def run_causal_discovery(algorithm_tag, data, config):
    algorithm = algorithm_tag.split("_")[0]
    if algorithm == "pc":
        G = run_pc(data, config)
    elif algorithm == "fci":
        G = run_fci(data, config)
    elif algorithm == "ges":
        G = run_ges(data, config)
    elif algorithm == "es":
        G = run_es(data, config)
    elif algorithm == "icalingam":
        G = run_icalingam(data, config)
    elif algorithm == "directlingam":
        G = run_directlingam(data, config)
    elif algorithm == "grasp":
        G = run_grasp(data, config)
    else:
        raise NotImplementedError(f"{algorithm} was not yet implemented")

    return G
