import logging
import sys
from contextlib import contextmanager
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GraphClass import CausalGraph, GeneralGraph
from causallearn.search.ScoreBased.GES import ges


@contextmanager
def redirect_stdout_to_logger():
    class LoggerWriter:
        def __init__(self):
            # Use the same logger as setup in setup_logging()
            self.logger = logging.getLogger(__name__)

        def write(self, message):
            if message and not message.isspace():
                self.logger.info(message.strip())

        def flush(self):
            pass

    stdout = sys.stdout
    sys.stdout = LoggerWriter()
    try:
        yield
    finally:
        sys.stdout = stdout


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
        with redirect_stdout_to_logger():
            cg, edges = fci(
                data,
                independence_test_method=config["independence_test_method"],
                alpha=config["alpha"],
                depth=config["depth"],
                max_path_length=config["max_path_length"],
                background_knowledge=config["background_knowledge"],
                verbose=True,
            )
    elif algorithm == "ges":
        required_params = ["score_func", "maxP", "parameters"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required config parameter: {param}")

        with redirect_stdout_to_logger():
            cg = ges(
                data,
                score_func=config["score_func"],
                maxP=config["maxP"],
                parameters=config["parameters"],
            )["G"]

    else:
        raise NotImplementedError(f"{algorithm} was not yet implemented")

    if isinstance(cg, CausalGraph):
        return cg.G
    elif isinstance(cg, GeneralGraph):
        return cg
    else:
        raise TypeError("Unexpected graph type")
