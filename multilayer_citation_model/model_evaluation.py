"""
Model evaluation and comparison utilities for the citation inference model.

Functions for model comparison using AIC/BIC, temporal validation,
and performance metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass

from .multilayer_network import MultilayerNetwork
from .inference import ModelConfig, OptimizationResult, optimize_parameters


class ModelComparison(NamedTuple):
    """Model comparison metrics."""
    model: str
    log_likelihood: float
    num_params: int
    aic: float
    bic: float
    optimal_params: List[float]


@dataclass
class ValidationResult:
    """Result from temporal validation."""
    optimal_params: List[float]
    log_likelihood: float
    success: bool
    iterations: int
    message: str
    time_window: Tuple[float, float]
    num_citations: int


def calculate_aic(log_likelihood: float, num_params: int) -> float:
    """Calculate Akaike Information Criterion."""
    return 2 * num_params - 2 * log_likelihood


def calculate_bic(
    log_likelihood: float, num_params: int, num_observations: int
) -> float:
    """Calculate Bayesian Information Criterion."""
    return np.log(num_observations) * num_params - 2 * log_likelihood


def compare_models(
    results: List["OptimizationResult"], model_names: List[str]
) -> pd.DataFrame:
    """
    Compare multiple models using AIC and BIC.

    Args:
        results: List of OptimizationResult from optimize_parameters
        model_names: Names for each model

    Returns:
        DataFrame with model comparison metrics
    """
    comparison_data = []

    for result, name in zip(results, model_names):
        log_likelihood = result.log_likelihood
        num_params = len(result.optimal_params)

        # Estimate number of observations (citations)
        # This should be passed in, but we'll use a default
        num_obs = 1000  # TODO: Pass this properly

        aic = calculate_aic(log_likelihood, num_params)
        bic = calculate_bic(log_likelihood, num_params, num_obs)

        comparison_data.append(
            ModelComparison(
                model=name,
                log_likelihood=log_likelihood,
                num_params=num_params,
                aic=aic,
                bic=bic,
                optimal_params=result.optimal_params,
            )
        )

    df = pd.DataFrame([comp._asdict() for comp in comparison_data])

    # Sort by AIC (lower is better)
    df = df.sort_values("aic")
    df["delta_aic"] = df["aic"] - df["aic"].min()

    return df


def validate_model_temporal_stability(
    network: MultilayerNetwork,
    citations: List[Tuple[int, int, float]],
    config: ModelConfig,
    time_windows: List[Tuple[float, float]],
) -> List["ValidationResult"]:
    """
    Validate model stability across different time windows.

    Args:
        network: Full MultilayerNetwork
        citations: All citations
        config: Model configuration
        time_windows: List of (start_time, end_time) tuples

    Returns:
        List of ValidationResult for each time window
    """
    # Vectorized temporal validation processing
    if not citations or not time_windows:
        return []

    # Convert citations to numpy array for efficient filtering
    citation_array = np.array(citations)
    citation_times = citation_array[:, 2].astype(float)

    results = []

    # Process all time windows
    for start_time, end_time in time_windows:
        # Vectorized citation filtering using boolean indexing
        time_mask = (citation_times >= start_time) & (citation_times <= end_time)
        window_citations = citation_array[time_mask].tolist()

        if len(window_citations) < 10:  # Skip if too few citations
            continue

        # Create network subset for this time window
        window_network = network.get_network_at_time(end_time)

        # Convert back to expected format
        window_citations = [(int(c), int(d), float(t)) for c, d, t in window_citations]

        # Optimize parameters for this window
        opt_result = optimize_parameters(window_network, window_citations, config)
        result = ValidationResult(
            optimal_params=opt_result.optimal_params,
            log_likelihood=opt_result.log_likelihood,
            success=opt_result.success,
            iterations=opt_result.iterations,
            message=opt_result.message,
            time_window=(start_time, end_time),
            num_citations=len(window_citations),
        )

        results.append(result)

    return results