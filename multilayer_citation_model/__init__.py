"""
Multilayer Citation Model Package

A Python package for modeling and predicting citations in multilayer networks,
combining citation dynamics with social network effects.

This package implements the models from:
"Multilayer network approach to modeling authorship influence on citation dynamics"

Main Components:
- MultilayerNetwork: Core data structure for citation and authorship networks
- CitationInference: Main inference engine for citation prediction
- CitationModels: Various citation constituent models (UNIF, PA, PA-RD, PA-NL-RD)
- SocialConstituents: Social network features (NAUT, NCOAUT, etc.)
- ModelConfig: Configuration for different model variants

Quick Start:
    >>> from multilayer_citation_model import MultilayerNetwork, CitationInference, ModelConfig
    >>> from multilayer_citation_model import generate_synthetic_network
    >>>
    >>> # Create a synthetic network
    >>> network = generate_synthetic_network(n_pubs=100, n_authors=50)
    >>>
    >>> # Configure model
    >>> config = ModelConfig("PA-RD", "NCOAUT", "ADDITIVE")
    >>>
    >>> # Create inference model
    >>> inference = CitationInference(network, [0.5, 0.0, 1.0, 1.0, 100.0], config)
    >>>
    >>> # Predict citations
    >>> predictions = inference.predict_citations(-1, 50.0, top_k=10)
"""

__version__ = "1.0.0"
__author__ = "Sadamori Kojaku et al."
__email__ = ""

# Core classes
from .multilayer_network import MultilayerNetwork
from .inference import (
    CitationInference,
    ModelConfig,
    infer_parameters,
    paper_time_step_sampling,
)
from .citation_models import CitationModels, CitationModelType
from .social_constituents import SocialConstituents, SocialConstituentType
from .synthetic_network import PaperSyntheticGenerator

# Utility functions
from .data_loading import (
    load_network_from_dataframe as build_network_from_dataframes,
    split_temporal_data,
)
from .inference import (
    optimize_parameters,
    optimize_parameters_minimal,
    calculate_model_likelihood,
    calculate_model_likelihood_fast,
    precompute_network_stats,
    PrecomputedStats,
    OptimizationResult,
    # Parameter mapping utilities
    get_required_parameters,
    get_parameter_indices,
    minimal_to_full_params,
    full_to_minimal_params,
    get_minimal_bounds,
    get_minimal_initial_params,
)
from .model_evaluation import (
    ModelComparison,
    compare_models,
    calculate_aic,
    calculate_bic,
)


__all__ = [
    # Core classes
    "MultilayerNetwork",
    "CitationInference",
    "ModelConfig",
    "CitationModels",
    "CitationModelType",
    "SocialConstituents",
    "SocialConstituentType",
    # Synthetic network generation
    "PaperSyntheticGenerator",
    "paper_time_step_sampling",
    # Inference functions
    "infer_parameters",
    # Utility functions
    "build_network_from_dataframes",
    "generate_synthetic_network",
    "split_temporal_data",
    "optimize_parameters",
    "optimize_parameters_fast",
    "optimize_parameters_minimal",
    "calculate_model_likelihood",
    "calculate_model_likelihood_fast",
    "precompute_network_stats",
    "PrecomputedStats",
    "OptimizationResult",
    "ModelComparison",
    "compare_models",
    "calculate_aic",
    "calculate_bic",
    # Parameter mapping utilities
    "get_required_parameters",
    "get_parameter_indices",
    "minimal_to_full_params",
    "full_to_minimal_params",
    "get_minimal_bounds",
    "get_minimal_initial_params",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Package-level configuration
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="divide by zero encountered"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered"
)


def get_version():
    """Get package version."""
    return __version__


def citation():
    """Get citation information for the package."""
    return """
    If you use this package in your research, please cite:

    [Citation information to be added - please include the relevant paper]

    BibTeX:
    @article{multilayer_citation_model,
        title={Multilayer network approach to modeling authorship influence on citation dynamics},
        author={[Authors to be added]},
        journal={[Journal to be added]},
        year={[Year to be added]},
        doi={[DOI to be added]}
    }
    """


def info():
    """Print package information."""
    print(f"Multilayer Citation Model v{__version__}")
    print(f"Authors: {__author__}")
    print("A Python package for multilayer citation network modeling and prediction")
    print(
        f"Documentation: https://github.com/skojaku/Legal-Citations/tree/main/libs/authorship_citation_model"
    )
