from typing import Any, Dict


TEST_METHODS: Dict[str, Dict[str, Any]] = {
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
}

SCORE_METHODS: Dict[str, Dict[str, Any]] = {
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

BOOTSTRAP_PRESETS: Dict[str, Dict[str, Any]] = {
    "bootstrap100": {
        "numberResampling": 10,
        "percent_resample_size": 100,
        "with_replacement": True,
        "add_original": True,
        "resampling_ensemble": 1,
        # "seed": 42,
    },
    "jackknife90": {
        "numberResampling": 10,
        "percent_resample_size": 90,
        "with_replacement": False,
        "add_original": True,
        "resampling_ensemble": 1,
        # "seed": 42,
    },
}
