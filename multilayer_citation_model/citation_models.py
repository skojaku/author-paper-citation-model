"""
Citation Constituent Models

Implementation of citation constituent models K_j from the paper:
- UNIF: Uniform attachment (baseline)
- PA: Preferential attachment
- PA-RD: Preferential attachment with relevance decay
- PA-NL-RD: Nonlinear preferential attachment with relevance decay

These models capture how likely a publication is to be cited based on its
citation history and age.
"""

import numpy as np
from typing import Optional
from enum import Enum
from .multilayer_network import MultilayerNetwork


class CitationModelType(Enum):
    """Enumeration of available citation model types."""
    UNIF = "UNIF"
    PA = "PA"
    PA_RD = "PA-RD"
    PA_NL_RD = "PA-NL-RD"


class CitationModels:
    """Collection of citation constituent models K_j."""

    @staticmethod
    def unif(
        network: MultilayerNetwork,
        pub_id: int,
        current_time: float,
        gamma: float = 1.0,
        delta: float = 1.0,
        tau: float = 1.0,
    ) -> float:
        """
        Uniform attachment: K_j = 1

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t
            gamma, delta, tau: Parameters (unused for UNIF)

        Returns:
            Citation constituent value (always 1.0)
        """
        return 1.0

    @staticmethod
    def pa(
        network: MultilayerNetwork,
        pub_id: int,
        current_time: float,
        gamma: float = 1.0,
        delta: float = 1.0,
        tau: float = 1.0,
    ) -> float:
        """
        Preferential attachment: K_j = k^pp,in_j(t) + δ

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t
            gamma: Unused for PA
            delta: Offset parameter
            tau: Unused for PA

        Returns:
            Citation constituent value
        """
        in_degree = network.get_in_degree(pub_id, current_time)
        return float(in_degree + delta)

    @staticmethod
    def pa_rd(
        network: MultilayerNetwork,
        pub_id: int,
        current_time: float,
        gamma: float = 1.0,
        delta: float = 1.0,
        tau: float = 1.0,
    ) -> float:
        """
        Preferential attachment with relevance decay: K_j = (k^pp,in_j(t) + δ) × e^(-(t-t_j)/τ)

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t
            gamma: Unused for PA-RD
            delta: Offset parameter
            tau: Decay time constant

        Returns:
            Citation constituent value
        """
        in_degree = network.get_in_degree(pub_id, current_time)
        pub_time = network.get_publication_time(pub_id)

        if pub_time is None:
            return 0.0

        age = current_time - pub_time
        decay_factor = np.exp(-age / tau) if tau > 0 else 1.0

        return float((in_degree + delta) * decay_factor)

    @staticmethod
    def pa_nl_rd(
        network: MultilayerNetwork,
        pub_id: int,
        current_time: float,
        gamma: float = 1.0,
        delta: float = 1.0,
        tau: float = 1.0,
    ) -> float:
        """
        Nonlinear preferential attachment with relevance decay: K_j = (k^pp,in_j(t) + 1)^γ × e^(-(t-t_j)/τ)

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t
            gamma: Nonlinearity exponent
            delta: Unused for PA-NL-RD (using 1 as offset)
            tau: Decay time constant

        Returns:
            Citation constituent value
        """
        in_degree = network.get_in_degree(pub_id, current_time)
        pub_time = network.get_publication_time(pub_id)

        if pub_time is None:
            return 0.0

        age = current_time - pub_time
        decay_factor = np.exp(-age / tau) if tau > 0 else 1.0

        # Use 1 as offset (k + 1) instead of delta for PA-NL-RD
        return float(((in_degree + 1) ** gamma) * decay_factor)

    @staticmethod
    def compute_citation_constituent(
        network: MultilayerNetwork,
        pub_id: int,
        current_time: float,
        model_type: str,
        gamma: float = 1.0,
        delta: float = 1.0,
        tau: float = 1.0,
    ) -> float:
        """
        Compute citation constituent K_j for a given model type.

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t
            model_type: One of "UNIF", "PA", "PA-RD", "PA-NL-RD"
            gamma: Nonlinearity exponent
            delta: Offset parameter
            tau: Decay time constant

        Returns:
            Citation constituent value K_j

        Raises:
            ValueError: If model_type is not recognized
        """
        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_enum = CitationModelType(model_type)
            except ValueError:
                available_types = [model.value for model in CitationModelType]
                raise ValueError(
                    f"Unknown citation model type: {model_type}. "
                    f"Must be one of {available_types}"
                )
        else:
            model_enum = model_type

        # Use match-case for dispatch
        match model_enum:
            case CitationModelType.UNIF:
                return CitationModels.unif(network, pub_id, current_time, gamma, delta, tau)
            case CitationModelType.PA:
                return CitationModels.pa(network, pub_id, current_time, gamma, delta, tau)
            case CitationModelType.PA_RD:
                return CitationModels.pa_rd(network, pub_id, current_time, gamma, delta, tau)
            case CitationModelType.PA_NL_RD:
                return CitationModels.pa_nl_rd(network, pub_id, current_time, gamma, delta, tau)
            case _:
                available_types = [model.value for model in CitationModelType]
                raise ValueError(
                    f"Unknown citation model type: {model_enum}. "
                    f"Must be one of {available_types}"
                )

    @staticmethod
    def compute_all_citation_constituents(
        network: MultilayerNetwork,
        current_time: float,
        model_type: str,
        gamma: float = 1.0,
        delta: float = 1.0,
        tau: float = 1.0,
    ) -> np.ndarray:
        """
        Compute citation constituents for all publications at current time.

        Args:
            network: MultilayerNetwork instance
            current_time: Current time t
            model_type: Citation model type
            gamma: Nonlinearity exponent
            delta: Offset parameter
            tau: Decay time constant

        Returns:
            numpy array of citation constituent values
        """
        available_pubs = network.get_publications_at_time(current_time)
        n_pubs = len(available_pubs)

        if n_pubs == 0:
            return np.array([])

        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_enum = CitationModelType(model_type)
            except ValueError:
                available_types = [model.value for model in CitationModelType]
                raise ValueError(f"Unknown citation model type: {model_type}. Must be one of {available_types}")
        else:
            model_enum = model_type

        # Vectorized computation based on model type
        match model_enum:
            case CitationModelType.UNIF:
                return np.ones(n_pubs, dtype=np.float64)

            case CitationModelType.PA:
                # Vectorized: Get in-degrees for all publications
                in_degrees = network.get_in_degrees(available_pubs, current_time)
                return in_degrees + delta

            case CitationModelType.PA_RD:
                # Vectorized: Get in-degrees and publication times
                in_degrees = network.get_in_degrees(available_pubs, current_time)
                pub_times = network.get_publication_times(available_pubs)

                ages = current_time - pub_times
                decay_factors = np.exp(-ages / tau) if tau > 0 else np.ones_like(ages)
                return (in_degrees + delta) * decay_factors

            case CitationModelType.PA_NL_RD:
                # Vectorized: Get in-degrees and publication times
                in_degrees = network.get_in_degrees(available_pubs, current_time)
                pub_times = network.get_publication_times(available_pubs)

                ages = current_time - pub_times
                decay_factors = np.exp(-ages / tau) if tau > 0 else np.ones_like(ages)
                return ((in_degrees + 1) ** gamma) * decay_factors

            case _:
                available_types = [model.value for model in CitationModelType]
                raise ValueError(f"Unknown citation model type: {model_enum}. Must be one of {available_types}")

    @staticmethod
    def get_available_models() -> np.ndarray:
        """Get array of available citation model types."""
        return np.array([model.value for model in CitationModelType])
