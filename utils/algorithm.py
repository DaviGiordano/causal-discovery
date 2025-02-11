from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GraphClass import CausalGraph, GeneralGraph


def run_causal_discovery(algorithm_tag, data, config):
    algorithm = algorithm_tag.split("_")[0]
    if algorithm == "pc":
        required_params = [
            "alpha",
            "indep_test",
            "stable",
            "uc_rule",
            "uc_priority",
            "mvpc",
            "correction_name",
            "background_knowledge",
        ]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required config parameter: {param}")

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
            verbose=False,
            show_progress=True,
        )
    elif algorithm == "fci":
        required_params = [
            "independence_test_method",
            "alpha",
            "depth",
            "max_path_length",
            "background_knowledge",
        ]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required config parameter: {param}")

        cg, edges = fci(
            data,
            independence_test_method=config["independence_test_method"],
            alpha=config["alpha"],
            depth=config["depth"],
            max_path_length=config["max_path_length"],
            background_knowledge=config["background_knowledge"],
        )

    else:
        raise NotImplementedError(f"{algorithm} was not yet implemented")

    if isinstance(cg, CausalGraph):
        return cg.G
    elif isinstance(cg, GeneralGraph):
        return cg
    else:
        raise TypeError("Unexpected graph type")
