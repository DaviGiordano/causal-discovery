from functools import wraps
import inspect

PC_REQUIRED_PARAMS = [
    "alpha",
    "indep_test",
    "stable",
    "uc_rule",
    "uc_priority",
    "mvpc",
    "correction_name",
    "background_knowledge",
]

FCI_REQUIRED_PARAMS = [
    "independence_test_method",
    "alpha",
    "depth",
    "max_path_length",
    "background_knowledge",
]

GES_REQUIRED_PARAMS = ["score_func", "maxP", "parameters"]

ES_REQUIRED_PARAMS = [
    "search_method",
    "use_path_extension",
    "use_k_cycle_heuristic",
    "k",
    "max_parents",
    "super_graph",
]

ICALINGAM_REQUIRED_PARAMS = ["random_state", "max_iter"]

DIRECTLINGAM_REQUIRED_PARAMS = [
    "random_state",
    "prior_knowledge",
    "apply_prior_knowledge_softly",
    "measure",
]

RCD_REQUIRED_PARAMS = [
    "max_explanatory_num",
    "cor_alpha",
    "ind_alpha",
    "shapiro_alpha",
    "MLHSICR",
    "bw_method",
]

GRASP_REQUIRED_PARAMS = ["score_func", "parameters", "depth"]


def validate_params(required_params):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if "config" not in bound_args.arguments:
                raise ValueError("No 'config' parameter provided to the function")

            config = bound_args.arguments.get("config")

            if config is None:
                raise ValueError("Config parameter is None")

            for param in required_params:
                if param not in config:
                    raise ValueError(f"Missing required config parameter: {param}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
