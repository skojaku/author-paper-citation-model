"""
Citation Inference and Optimization Engine

Combined module containing:
1. Core citation probability calculation and inference functions
2. Parameter optimization and likelihood calculation
3. Pre-computation utilities for efficient optimization

Main components:
- CitationInference: Core inference engine with probability calculations
- ModelConfig: Configuration for citation model components
- OptimizationResult: Results from parameter optimization
- Parameter optimization functions with numba acceleration
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.optimize import minimize

from .multilayer_network import MultilayerNetwork
from .citation_models import CitationModels
from .social_constituents import SocialConstituents

# Optional numba import for acceleration
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args else decorator


@dataclass
class ModelConfig:
    """Configuration for citation model components."""

    citation_type: str  # "UNIF", "PA", "PA-RD", "PA-NL-RD"
    social_type: Optional[str] = (
        None  # "NAUT", "NCOAUT", "MAXCOAUT", "NPUB", "MAXPUB", or None
    )
    coupling_type: str = "NONE"  # "ADDITIVE", "MULTIPLICATIVE", "NONE"


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""

    optimal_params: List[float]
    log_likelihood: float
    success: bool
    iterations: int
    message: str


@dataclass
class PrecomputedStats:
    """Pre-computed network statistics for fast likelihood calculation."""

    unique_times: np.ndarray
    edges: np.ndarray  # (N, 2) array of [src, trg] edges sorted by src (time)
    indptr: np.ndarray  # Index pointers for O(1) edge filtering by time
    time_pub_to_social: dict  # (time, pub) -> social constituent


# Parameter Management Functions


def get_required_parameters(config: ModelConfig) -> List[str]:
    """
    Get the minimal set of parameters required for a given model configuration.

    Args:
        config: ModelConfig instance

    Returns:
        List of parameter names that are actually used by this configuration
    """
    required = []

    # Citation model parameters
    if config.citation_type == "PA":
        required.append("delta")
    elif config.citation_type == "PA-RD":
        required.extend(["delta", "tau"])
    elif config.citation_type == "PA-NL-RD":
        required.extend(["gamma", "tau"])
    # UNIF requires no parameters

    # Social and coupling parameters
    if config.social_type is not None:
        if config.coupling_type == "ADDITIVE":
            required.append("alpha")
        elif config.coupling_type == "MULTIPLICATIVE":
            required.append("beta")

    return required


def get_parameter_indices(required_params: List[str]) -> List[int]:
    """
    Get the indices of required parameters in the full parameter vector.

    Args:
        required_params: List of parameter names

    Returns:
        List of indices [0=alpha, 1=beta, 2=gamma, 3=delta, 4=tau]
    """
    param_map = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3, "tau": 4}
    return [param_map[param] for param in required_params]


def minimal_to_full_params(
    minimal_params: np.ndarray,
    required_indices: List[int],
    default_values: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Convert minimal parameter vector to full 5-parameter vector.

    Args:
        minimal_params: The minimal parameter values being optimized
        required_indices: Indices where these parameters go in full vector
        default_values: Default values for unused parameters (if None, uses standard defaults)

    Returns:
        Full 5-parameter vector [alpha, beta, gamma, delta, tau]
    """
    if default_values is None:
        default_values = [0.5, 0.0, 1.0, 1.0, 100.0]  # [alpha, beta, gamma, delta, tau]

    full_params = np.array(default_values, dtype=np.float64)
    full_params[required_indices] = minimal_params
    return full_params


def full_to_minimal_params(
    full_params: np.ndarray, required_indices: List[int]
) -> np.ndarray:
    """
    Extract minimal parameter subset from full parameter vector.

    Args:
        full_params: Full 5-parameter vector
        required_indices: Indices of parameters to extract

    Returns:
        Minimal parameter vector
    """
    return full_params[required_indices]


def get_minimal_bounds(
    config: ModelConfig, full_bounds: Optional[List[Tuple[float, float]]] = None
) -> List[Tuple[float, float]]:
    """
    Get parameter bounds for only the required parameters.

    Args:
        config: ModelConfig instance
        full_bounds: Full bounds list (if None, uses standard bounds)

    Returns:
        Minimal bounds list for required parameters only
    """
    if full_bounds is None:
        full_bounds = [
            (0.01, 0.99),  # α: mixture weight
            (-5.0, 5.0),  # β: social exponent
            (0.1, 5.0),  # γ: citation exponent
            (0.1, 10.0),  # δ: offset parameter (relax lower bound)
            (0.1, 10000.0),  # τ: decay time constant
        ]

    required_params = get_required_parameters(config)
    required_indices = get_parameter_indices(required_params)

    return [full_bounds[i] for i in required_indices]


def get_minimal_initial_params(
    config: ModelConfig, full_initial: Optional[List[float]] = None
) -> List[float]:
    """
    Get initial parameter values for only the required parameters.

    Args:
        config: ModelConfig instance
        full_initial: Full initial parameter list (if None, uses standard values)

    Returns:
        Minimal initial parameter list for required parameters only
    """
    if full_initial is None:
        full_initial = [
            0.7,
            0.0,
            1.0,
            1.2,
            100.0,
        ]  # [alpha, beta, gamma, delta, tau] - use values closer to true parameters

    required_params = get_required_parameters(config)
    required_indices = get_parameter_indices(required_params)

    return [full_initial[i] for i in required_indices]


class CitationInference:
    """Main citation inference engine."""

    def __init__(
        self, network: MultilayerNetwork, theta: np.ndarray, config: ModelConfig
    ):
        """
        Initialize citation inference model.

        Args:
            network: MultilayerNetwork instance
            theta: Model parameters [α, β, γ, δ, τ]
            config: Model configuration
        """
        self.network = network
        self.theta = np.asarray(theta, dtype=np.float64)
        self.config = config

        # Extract parameters with defaults
        params = np.pad(self.theta, (0, max(0, 5 - len(self.theta))), constant_values=0)
        params[0] = params[0] if len(self.theta) > 0 else 0.5  # alpha
        params[2] = params[2] if len(self.theta) > 2 else 1.0  # gamma
        params[3] = params[3] if len(self.theta) > 3 else 1.0  # delta
        params[4] = params[4] if len(self.theta) > 4 else 1.0  # tau

        self.alpha, self.beta, self.gamma, self.delta, self.tau = params

    def citation_probability(
        self, citing_pub: int, cited_pub: int, current_time: float
    ) -> float:
        """
        Calculate citation probability π^pp_ij using Equation 1.1.

        Args:
            citing_pub: ID of citing publication i
            cited_pub: ID of cited publication j
            current_time: Current time t

        Returns:
            Citation probability π^pp_ij
        """
        # Get all available publications
        available_pubs = self.network.get_publications_at_time(current_time)

        if len(available_pubs) == 0:
            return 0.0

        # Find index of cited publication
        cited_idx = np.where(available_pubs == cited_pub)[0]
        if len(cited_idx) == 0:
            return 0.0
        cited_idx = cited_idx[0]

        # Calculate all citation constituents vectorized
        K_all = CitationModels.compute_all_citation_constituents(
            self.network,
            current_time,
            self.config.citation_type,
            self.gamma,
            self.delta,
            self.tau,
        )

        if len(K_all) == 0:
            return 0.0

        K_j = K_all[cited_idx]

        # Calculate social constituents if needed
        if self.config.social_type is not None:
            eta_all = SocialConstituents.compute_all_social_constituents(
                self.network, current_time, self.config.social_type
            )
            if len(eta_all) == 0:
                return 0.0
            eta_j = eta_all[cited_idx]
        else:
            eta_all = np.ones_like(K_all)
            eta_j = 1.0

        # Calculate probability based on coupling type using vectorized operations
        if self.config.coupling_type == "ADDITIVE":
            # π^pp_ij = α × [K_j / ∑_l K_l] + (1-α) × [η_j / ∑_l η_l]
            denom_citation = np.sum(K_all)
            denom_social = np.sum(eta_all)

            if denom_citation > 0 and denom_social > 0:
                prob = self.alpha * (K_j / denom_citation) + (1 - self.alpha) * (
                    eta_j / denom_social
                )
            else:
                prob = 0.0

        elif self.config.coupling_type == "MULTIPLICATIVE":
            # π^pp_ij = [η_j^β × K_j] / [∑_l η_l^β × K_l]
            multiplicative_terms = (eta_all**self.beta) * K_all
            denom_multiplicative = np.sum(multiplicative_terms)

            if denom_multiplicative > 0:
                prob = ((eta_j**self.beta) * K_j) / denom_multiplicative
            else:
                prob = 0.0

        else:  # No coupling (citation only)
            denom_citation = np.sum(K_all)
            if denom_citation > 0:
                prob = K_j / denom_citation
            else:
                prob = 0.0

        return float(prob)

    def predict_citations(
        self, citing_pub: int, current_time: float, top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict which publications a new publication will cite.

        Args:
            citing_pub: ID of citing publication
            current_time: Current time
            top_k: Return only top k predictions (None for all)

        Returns:
            Array of (publication_id, probability) pairs sorted by probability (descending)
        """
        available_pubs = self.network.get_publications_at_time(current_time)

        if len(available_pubs) == 0:
            return np.array([])

        # Remove self-citation
        valid_pubs = available_pubs[available_pubs != citing_pub]

        if len(valid_pubs) == 0:
            return np.array([])

        # Calculate all citation constituents vectorized
        K_all = CitationModels.compute_all_citation_constituents(
            self.network,
            current_time,
            self.config.citation_type,
            self.gamma,
            self.delta,
            self.tau,
        )

        # Calculate social constituents if needed
        if self.config.social_type is not None:
            eta_all = SocialConstituents.compute_all_social_constituents(
                self.network, current_time, self.config.social_type
            )
        else:
            eta_all = np.ones_like(K_all)

        # Calculate probabilities vectorized
        if self.config.coupling_type == "ADDITIVE":
            denom_citation = np.sum(K_all)
            denom_social = np.sum(eta_all)

            if denom_citation > 0 and denom_social > 0:
                probs = self.alpha * (K_all / denom_citation) + (1 - self.alpha) * (
                    eta_all / denom_social
                )
            else:
                probs = np.zeros_like(K_all)

        elif self.config.coupling_type == "MULTIPLICATIVE":
            multiplicative_terms = (eta_all**self.beta) * K_all
            denom_multiplicative = np.sum(multiplicative_terms)

            if denom_multiplicative > 0:
                probs = multiplicative_terms / denom_multiplicative
            else:
                probs = np.zeros_like(K_all)

        else:  # No coupling (citation only)
            denom_citation = np.sum(K_all)
            if denom_citation > 0:
                probs = K_all / denom_citation
            else:
                probs = np.zeros_like(K_all)

        # Filter out self-citation and create result array
        valid_mask = available_pubs != citing_pub
        valid_probs = probs[valid_mask]
        valid_pub_ids = available_pubs[valid_mask]

        # Sort by probability (highest first)
        sort_indices = np.argsort(valid_probs)[::-1]

        if top_k is not None:
            sort_indices = sort_indices[:top_k]

        # Return as structured array for efficiency
        result = np.column_stack(
            (valid_pub_ids[sort_indices], valid_probs[sort_indices])
        )
        return result

    def sample_citations(
        self,
        citing_pub: int,
        current_time: float,
        num_citations: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample citations based on the probability distribution.

        Args:
            citing_pub: ID of citing publication
            current_time: Current time
            num_citations: Number of citations to sample
            seed: Random seed for reproducibility

        Returns:
            Array of sampled publication IDs
        """
        if seed is not None:
            np.random.seed(seed)

        available_pubs = self.network.get_publications_at_time(current_time)

        # Remove self-citation
        valid_pubs = available_pubs[available_pubs != citing_pub]

        if len(valid_pubs) == 0:
            return np.array([], dtype=np.int32)

        # Get prediction probabilities (reuse vectorized computation)
        predictions = self.predict_citations(citing_pub, current_time)

        if len(predictions) == 0:
            return np.array([], dtype=np.int32)

        # Extract probabilities and normalize
        probabilities = predictions[:, 1].astype(np.float64)
        pub_ids = predictions[:, 0].astype(np.int32)

        total_prob = np.sum(probabilities)
        if total_prob == 0:
            # Uniform sampling if all probabilities are zero
            probabilities = np.ones(len(probabilities)) / len(probabilities)
        else:
            probabilities = probabilities / total_prob

        # Sample without replacement
        num_citations = min(num_citations, len(pub_ids))
        sampled_indices = np.random.choice(
            len(pub_ids), size=num_citations, replace=False, p=probabilities
        )

        return pub_ids[sampled_indices]

    def predict_citation_count(
        self,
        target_pub: int,
        current_time: float,
        future_time: float,
        time_step: float = 1.0,
    ) -> float:
        """
        Predict how many citations a publication will receive in a time period.

        Args:
            target_pub: ID of publication to predict citations for
            current_time: Current time
            future_time: End time for prediction
            time_step: Time step size for discrete prediction

        Returns:
            Expected number of citations
        """
        total_citations = 0.0

        # Discrete time steps
        t = current_time
        while t < future_time:
            # Assume one new publication added at each time step
            # This is a simplification; in practice you'd need to model the arrival process
            prob = self.citation_probability(
                -1, target_pub, t
            )  # Use -1 as dummy new pub ID
            total_citations += prob
            t += time_step

        return total_citations

    def get_citation_probabilities_matrix(
        self, current_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get full citation probability matrix for all publication pairs.

        Args:
            current_time: Current time

        Returns:
            (probability_matrix, publication_ids) where matrix[i,j] = P(i cites j)
        """
        available_pubs = self.network.get_publications_at_time(current_time)
        n_pubs = len(available_pubs)

        if n_pubs == 0:
            return np.array([]), np.array([])

        # Calculate constituents once for all publications
        K_all = CitationModels.compute_all_citation_constituents(
            self.network,
            current_time,
            self.config.citation_type,
            self.gamma,
            self.delta,
            self.tau,
        )

        if self.config.social_type is not None:
            eta_all = SocialConstituents.compute_all_social_constituents(
                self.network, current_time, self.config.social_type
            )
        else:
            eta_all = np.ones_like(K_all)

        # Calculate probability matrix vectorized
        prob_matrix = np.zeros((n_pubs, n_pubs))

        if self.config.coupling_type == "ADDITIVE":
            denom_citation = np.sum(K_all)
            denom_social = np.sum(eta_all)

            if denom_citation > 0 and denom_social > 0:
                citation_probs = K_all / denom_citation
                social_probs = eta_all / denom_social

                # Broadcast to create full probability matrix
                prob_matrix = (
                    self.alpha * citation_probs[np.newaxis, :]
                    + (1 - self.alpha) * social_probs[np.newaxis, :]
                )

        elif self.config.coupling_type == "MULTIPLICATIVE":
            multiplicative_terms = (eta_all**self.beta) * K_all
            denom_multiplicative = np.sum(multiplicative_terms)

            if denom_multiplicative > 0:
                prob_matrix = multiplicative_terms[np.newaxis, :] / denom_multiplicative

        else:  # No coupling (citation only)
            denom_citation = np.sum(K_all)
            if denom_citation > 0:
                prob_matrix = K_all[np.newaxis, :] / denom_citation

        # Set diagonal to zero (no self-citations)
        np.fill_diagonal(prob_matrix, 0)

        return prob_matrix, available_pubs

    def evaluate_model(
        self, test_citations: np.ndarray, current_time: float
    ) -> np.ndarray:
        """
        Evaluate model performance on test citations.

        Args:
            test_citations: Array of (citing_pub, cited_pub, time) tuples
            current_time: Current time for evaluation

        Returns:
            Array of evaluation metrics [log_likelihood, mean_probability, num_citations]
        """
        if len(test_citations) == 0:
            return np.array([0.0, 0.0, 0.0])

        # Filter citations by time
        test_citations = np.asarray(test_citations)
        valid_mask = test_citations[:, 2] <= current_time
        valid_citations = test_citations[valid_mask]

        if len(valid_citations) == 0:
            return np.array([0.0, 0.0, 0.0])

        # Vectorized probability calculation
        citing_pubs = valid_citations[:, 0].astype(int)
        cited_pubs = valid_citations[:, 1].astype(int)

        predictions = np.array(
            [
                self.citation_probability(citing_pub, cited_pub, current_time)
                for citing_pub, cited_pub in zip(citing_pubs, cited_pubs)
            ]
        )

        # Calculate metrics
        log_likelihood = np.sum(np.log(predictions + 1e-15))
        mean_probability = np.mean(predictions)
        num_citations = len(predictions)

        return np.array([log_likelihood, mean_probability, num_citations])


# Parameter Optimization Functions


def precompute_network_stats(
    network: MultilayerNetwork,
    citations: List[Tuple[int, int, float]],
    config: ModelConfig,
) -> PrecomputedStats:
    """
    Pre-compute network statistics for efficient likelihood calculation.
    Converts citation matrix to sorted edge list for fast in-degree computation.

    Args:
        network: MultilayerNetwork instance
        citations: List of (citing_pub, cited_pub, time) tuples
        config: Model configuration

    Returns:
        PrecomputedStats object with pre-computed values
    """
    # Get all unique times from citations
    unique_times = np.unique([time for _, _, time in citations])

    # Convert citation matrix to sorted edge list format
    citation_matrix = network.citation_matrix
    src_indices, trg_indices = citation_matrix.nonzero()
    edges = np.column_stack([src_indices, trg_indices])

    # Sort edges by source (time) for efficient filtering
    sort_idx = np.argsort(edges[:, 0])
    edges = edges[sort_idx]

    # Create index pointer array for O(1) time-based edge filtering
    max_time = citation_matrix.shape[0] if len(edges) > 0 else 0
    indptr = np.zeros(max_time + 2, dtype=np.int32)

    if len(edges) > 0:
        # Find first occurrence of each src time in sorted edges
        src_times = edges[:, 0]
        unique_src_times, first_indices = np.unique(src_times, return_index=True)

        # Fill indptr: indptr[t] = first index where edges[i, 0] >= t
        for src_time, first_idx in zip(unique_src_times, first_indices):
            indptr[src_time] = first_idx

        # Fill gaps: if time t has no edges, indptr[t] = indptr[t+1]
        for t in range(max_time, -1, -1):
            if indptr[t] == 0 and t < max_time:
                indptr[t] = indptr[t + 1]

        # Set final sentinel value
        indptr[-1] = len(edges)

    time_pub_to_social = {}

    # Only pre-compute social constituents if needed (expensive computation)
    if config.social_type is not None:
        for time in unique_times:
            # Get available publications at this time (pub_id <= time)
            available_pubs = np.arange(int(time) + 1)

            if len(available_pubs) == 0:
                continue

            social_constituents = SocialConstituents.compute_all_social_constituents(
                network, time, config.social_type
            )
            for pub, social_const in zip(available_pubs, social_constituents):
                # Ensure social constituents are positive and non-zero
                time_pub_to_social[(time, pub)] = max(social_const, 1e-6)

    return PrecomputedStats(
        unique_times=unique_times,
        edges=edges,
        indptr=indptr,
        time_pub_to_social=time_pub_to_social,
    )


@njit(cache=True)
def _compute_citation_constituents_numba(
    edges: np.ndarray,
    indptr: np.ndarray,
    available_pubs: np.ndarray,
    current_time: float,
    citation_type: int,  # 0=UNIF, 1=PA, 2=PA-RD, 3=PA-NL-RD
    delta: float,
    gamma: float,
    tau: float,
) -> np.ndarray:
    """
    Numba-accelerated computation of citation constituents.

    Args:
        edges: (N, 2) array of [src, trg] edges sorted by src
        indptr: Index pointers for edge filtering
        available_pubs: Available publications at current time
        current_time: Current citation time
        citation_type: Citation model type (0-3)
        delta, gamma, tau: Model parameters

    Returns:
        Citation constituents K_all for all available publications
    """
    n_pubs = len(available_pubs)
    K_all = np.zeros(n_pubs, dtype=np.float64)

    if citation_type == 0:  # UNIF
        K_all[:] = 1.0
        return K_all

    # O(1) edge filtering using pre-computed index pointers
    current_time_int = int(current_time)
    if current_time_int < len(indptr):
        n_valid_edges = indptr[current_time_int]
    else:
        n_valid_edges = len(edges)

    # Calculate in-degrees using bincount equivalent
    in_degrees = np.zeros(n_pubs, dtype=np.int32)
    for i in range(n_valid_edges):
        trg = edges[i, 1]
        if trg < n_pubs:  # Only count edges targeting available publications
            in_degrees[trg] += 1

    # Compute citation constituents based on model type
    for i in range(n_pubs):
        pub = available_pubs[i]
        in_degree = in_degrees[i]

        if citation_type == 1:  # PA
            K_all[i] = in_degree + delta
        elif citation_type == 2:  # PA-RD
            pub_time = pub
            age = current_time - pub_time
            decay_factor = np.exp(-age / tau) if tau > 0 else 1.0
            K_all[i] = (in_degree + delta) * decay_factor
        elif citation_type == 3:  # PA-NL-RD
            pub_time = pub
            age = current_time - pub_time
            decay_factor = np.exp(-age / tau) if tau > 0 else 1.0
            K_all[i] = ((in_degree + 1) ** gamma) * decay_factor

    return K_all


@njit(cache=True)
def _compute_coupling_probability_numba(
    K_all: np.ndarray,
    eta_all: np.ndarray,
    cited_idx: int,
    coupling_type: int,  # 0=NONE, 1=ADDITIVE, 2=MULTIPLICATIVE
    alpha: float,
    beta: float,
) -> float:
    """
    Numba-accelerated coupling probability computation.

    Args:
        K_all: Citation constituents for all available publications
        eta_all: Social constituents for all available publications
        cited_idx: Index of cited publication
        coupling_type: Coupling model type (0-2)
        alpha, beta: Model parameters

    Returns:
        Citation probability for the cited publication
    """
    K_j = K_all[cited_idx]
    eta_j = eta_all[cited_idx]

    if coupling_type == 0:  # No coupling (citation only)
        denom_citation = np.sum(K_all)
        if denom_citation > 0:
            return K_j / denom_citation
        else:
            return 0.0

    elif coupling_type == 1:  # ADDITIVE
        denom_citation = np.sum(K_all)
        denom_social = np.sum(eta_all)
        if denom_citation > 0 and denom_social > 0:
            return alpha * (K_j / denom_citation) + (1 - alpha) * (eta_j / denom_social)
        else:
            return 0.0

    elif coupling_type == 2:  # MULTIPLICATIVE
        if beta == 0.0:
            # When beta=0, eta^beta = 1 for all eta > 0
            denom_multiplicative = np.sum(K_all)
            if denom_multiplicative > 0:
                return K_j / denom_multiplicative
            else:
                return 0.0
        else:
            # Compute eta^beta * K for all publications
            multiplicative_terms = np.zeros(len(K_all))
            for i in range(len(K_all)):
                eta_clipped = max(eta_all[i], 1e-10)
                multiplicative_terms[i] = (eta_clipped**beta) * K_all[i]

            denom_multiplicative = np.sum(multiplicative_terms)
            if denom_multiplicative > 0:
                eta_j_clipped = max(eta_j, 1e-10)
                return ((eta_j_clipped**beta) * K_j) / denom_multiplicative
            else:
                return 0.0

    return 0.0


def calculate_model_likelihood_fast(
    theta: np.ndarray,
    network: MultilayerNetwork,
    precomputed: PrecomputedStats,
    citations: List[Tuple[int, int, float]],
    config: ModelConfig,
) -> float:
    """
    Ultra-fast likelihood calculation with numba acceleration.

    Args:
        theta: Model parameters [α, β, γ, δ, τ]
        network: MultilayerNetwork instance
        precomputed: Pre-computed network statistics (only social constituents)
        citations: List of (citing_pub, cited_pub, time) tuples
        config: Model configuration

    Returns:
        Negative log-likelihood
    """
    if not citations:
        return 0.0

    # Extract parameters with defaults
    params = np.pad(theta, (0, max(0, 5 - len(theta))), constant_values=0)
    alpha = params[0] if len(theta) > 0 else 0.5
    beta = params[1] if len(theta) > 1 else 0.0
    gamma = params[2] if len(theta) > 2 else 1.0
    delta = params[3] if len(theta) > 3 else 1.0
    tau = params[4] if len(theta) > 4 else 1.0

    # Convert config strings to integers for numba
    citation_type_map = {"UNIF": 0, "PA": 1, "PA-RD": 2, "PA-NL-RD": 3}
    coupling_type_map = {None: 0, "ADDITIVE": 1, "MULTIPLICATIVE": 2}

    citation_type = citation_type_map.get(config.citation_type, 0)
    coupling_type = coupling_type_map.get(config.coupling_type, 0)

    log_likelihood = 0.0

    for _, cited_pub, current_time in citations:
        # Calculate available publications on the fly (pub_id <= current_time)
        available_pubs = np.arange(int(current_time) + 1)

        if len(available_pubs) == 0:
            continue

        # Find index of cited publication
        cited_indices = np.where(available_pubs == cited_pub)[0]
        if len(cited_indices) == 0:
            continue
        cited_idx = cited_indices[0]

        # Use numba-accelerated citation constituent calculation
        K_all = _compute_citation_constituents_numba(
            precomputed.edges,
            precomputed.indptr,
            available_pubs,
            current_time,
            citation_type,
            delta,
            gamma,
            tau,
        )

        # Get social constituents
        if config.social_type is not None:
            eta_all = np.array(
                [
                    precomputed.time_pub_to_social.get((current_time, pub), 1.0)
                    for pub in available_pubs
                ]
            )
        else:
            eta_all = np.ones_like(K_all)

        # Use numba-accelerated coupling probability calculation
        prob = _compute_coupling_probability_numba(
            K_all, eta_all, cited_idx, coupling_type, alpha, beta
        )

        # Add to log likelihood (with epsilon to avoid log(0))
        prob = max(prob, 1e-32)
        log_likelihood += np.log(prob)

    return -log_likelihood


def calculate_model_likelihood(
    theta: np.ndarray,
    network: MultilayerNetwork,
    citations: List[Tuple[int, int, float]],
    config: ModelConfig,
) -> float:
    """
    Calculate negative log-likelihood for parameter optimization.

    Args:
        theta: Model parameters [α, β, γ, δ, τ]
        network: MultilayerNetwork instance
        citations: List of (citing_pub, cited_pub, time) tuples
        config: Model configuration

    Returns:
        Negative log-likelihood
    """
    inference = CitationInference(network, theta.tolist(), config)

    # Vectorized likelihood calculation
    if not citations:
        return 0.0

    # Extract citation data for vectorized processing
    citation_array = np.array(citations)
    citing_pubs = citation_array[:, 0].astype(int)
    cited_pubs = citation_array[:, 1].astype(int)
    cite_times = citation_array[:, 2].astype(float)

    # Calculate probabilities for all citations
    probs = np.array(
        [
            inference.citation_probability(citing, cited, time)
            for citing, cited, time in zip(citing_pubs, cited_pubs, cite_times)
        ]
    )

    # Apply epsilon to avoid log(0) and sum log probabilities
    probs = np.maximum(probs, 1e-15)
    log_likelihood = np.sum(np.log(probs))

    return -log_likelihood  # Return negative for minimization


def optimize_parameters_minimal(
    network: MultilayerNetwork,
    citations: List[Tuple[int, int, float]],
    config: ModelConfig,
    initial_params: Optional[List[float]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    use_fast: bool = True,
) -> OptimizationResult:
    """
    Optimized parameter optimization that only optimizes the minimal required parameters.

    This function significantly improves optimization efficiency by only optimizing
    the parameters that are actually used by the given model configuration, rather
    than optimizing all 5 parameters regardless of model type.

    Args:
        network: MultilayerNetwork instance
        citations: List of (citing_pub, cited_pub, time) tuples
        config: Model configuration
        initial_params: Initial parameter values for ALL parameters (will extract minimal subset)
        bounds: Parameter bounds for ALL parameters (will extract minimal subset)
        use_fast: Whether to use fast likelihood calculation

    Returns:
        OptimizationResult with optimal_params containing FULL 5-parameter vector
    """
    # Get minimal parameter configuration
    required_params = get_required_parameters(config)
    required_indices = get_parameter_indices(required_params)

    # Handle case where no parameters are needed (e.g., UNIF with no social component)
    if len(required_params) == 0:
        # No optimization needed, just calculate likelihood with default parameters
        default_full_params = np.array([0.5, 0.0, 1.0, 1.0, 100.0])
        if use_fast:
            precomputed = precompute_network_stats(network, citations, config)
            ll = calculate_model_likelihood_fast(
                default_full_params, network, precomputed, citations, config
            )
        else:
            ll = calculate_model_likelihood(
                default_full_params, network, citations, config
            )

        return OptimizationResult(
            optimal_params=default_full_params.tolist(),
            log_likelihood=-ll,
            success=True,
            iterations=0,
            message="No parameters to optimize",
        )

    # Get minimal bounds and initial parameters
    minimal_bounds = get_minimal_bounds(config, bounds)
    minimal_initial = get_minimal_initial_params(config, initial_params)

    # Pre-compute network statistics if using fast mode
    if use_fast:
        precomputed = precompute_network_stats(network, citations, config)

    # Create wrapper objective function that converts minimal to full parameters
    def objective_function(minimal_params):
        """Wrapper that converts minimal parameters to full vector for likelihood calculation."""
        full_params = minimal_to_full_params(minimal_params, required_indices)
        if use_fast:
            return calculate_model_likelihood_fast(
                full_params, network, precomputed, citations, config
            )
        else:
            return calculate_model_likelihood(full_params, network, citations, config)

    try:
        # Optimize only the minimal parameters
        result = minimize(
            objective_function,
            minimal_initial,
            method="L-BFGS-B",
            bounds=minimal_bounds,
            options={"maxiter": 100},
        )

        # Convert minimal optimal parameters back to full parameter vector
        full_optimal_params = minimal_to_full_params(result.x, required_indices)

        return OptimizationResult(
            optimal_params=full_optimal_params.tolist(),
            log_likelihood=-result.fun,
            success=result.success,
            iterations=result.nit,
            message=result.message,
        )

    except Exception as e:
        # Return failed result with default parameters
        default_full_params = [0.5, 0.0, 1.0, 1.0, 100.0]
        return OptimizationResult(
            optimal_params=default_full_params,
            log_likelihood=-np.inf,
            success=False,
            iterations=0,
            message=f"Optimization failed: {str(e)}",
        )


def optimize_parameters(
    network: MultilayerNetwork,
    citations: List[Tuple[int, int, float]],
    config: ModelConfig,
    initial_params: Optional[List[float]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    use_fast: bool = True,
) -> OptimizationResult:
    """
    Optimize model parameters using maximum likelihood estimation.

    Args:
        network: MultilayerNetwork instance
        citations: Training citations
        config: Model configuration
        initial_params: Initial parameter values
        bounds: Parameter bounds for optimization
        use_fast: Whether to use fast pre-computed version (recommended)

    Returns:
        OptimizationResult with optimization results
    """
    return optimize_parameters_minimal(
        network, citations, config, initial_params, bounds, use_fast
    )


def infer_parameters(
    network: MultilayerNetwork,
    citations: list,
    sample_size: int,
    config: ModelConfig = ModelConfig("PA", "NAUT", "ADDITIVE"),
    initial_params: list = None,
    bounds: list = None,
    seed: int = 42,
    use_fast: bool = True,
) -> dict:
    """
    Convenient function for parameter inference that takes a network, sample_size,
    and returns the parameter estimate.

    Args:
        network: MultilayerNetwork instance
        citations: List of (citing, cited, time) tuples
        sample_size: Number of citations to sample for inference
        config: Model configuration (default: PA with NAUT social constituent)
        initial_params: Initial parameter values [α, β, γ, δ, τ]
        bounds: Parameter bounds [(min, max), ...]
        seed: Random seed for sampling

    Returns:
        Dictionary with parameter estimates and inference results
    """
    # Set default parameters if not provided
    if initial_params is None:
        initial_params = [0.7, 0.0, 1.0, 1.2, 100.0]

    if bounds is None:
        bounds = [
            (0.1, 0.9),  # α: mixture weight
            (-5.0, 5.0),  # β: social exponent
            (0.1, 5.0),  # γ: citation exponent
            (0.1, 5.0),  # δ: offset parameter
            (10.0, 10000.0),  # τ: decay time constant
        ]

    # Sample citations using paper's methodology
    np.random.seed(seed)
    T = network.n_pubs
    sampled_citations = paper_time_step_sampling(citations, sample_size, T)

    # Perform parameter optimization (with minimal parameter optimization enabled)
    result = optimize_parameters(
        network,
        sampled_citations,
        config,
        initial_params=initial_params,
        bounds=bounds,
        use_fast=use_fast,
    )

    # Format results
    parameter_names = ["alpha", "beta", "gamma", "delta", "tau"]
    results = {
        "success": result.success,
        "log_likelihood": result.log_likelihood,
        "sample_size": len(sampled_citations),
        "original_citations": len(citations),
        "config": {
            "citation_type": config.citation_type,
            "social_type": config.social_type,
            "coupling_type": config.coupling_type,
        },
    }

    if result.success:
        # Add parameter estimates
        for _, (name, value) in enumerate(zip(parameter_names, result.optimal_params)):
            results[name] = value

        results["message"] = "Parameter inference completed successfully"
    else:
        results["message"] = f"Optimization failed: {result.message}"
        # Add NaN values for failed optimization
        for name in parameter_names:
            results[name] = float("nan")

    return results


def paper_time_step_sampling(
    citations: List[Tuple[int, int, float]], sample_size: int, T: int
) -> List[Tuple[int, int, float]]:
    """
    Implement the paper's time-step sampling methodology from Section III.C.

    From paper: "we sample, without replacement, a number S < T of time steps
    uniformly at random, {ts}s∈[1,S]. This set of time stamps uniformly covers
    the full time range [1, T] of the network growth"

    Args:
        citations: List of (citing, cited, time) tuples
        sample_size: Number of time steps to sample
        T: Total number of time steps

    Returns:
        List of citations that occur at the sampled time steps
    """
    if len(citations) == 0:
        return []

    # Get all unique time steps from the citation process
    all_times = sorted(set(time for _, _, time in citations))
    sampled_times = np.random.randint(1, T, sample_size)

    # Get all citations that happen at the sampled time steps
    sampled_citations = [
        (citing, cited, time)
        for citing, cited, time in citations
        if time in sampled_times
    ]

    return sampled_citations
