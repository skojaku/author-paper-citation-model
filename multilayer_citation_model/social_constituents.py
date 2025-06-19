"""
Social Constituent Models

Implementation of social constituent models η_j from the paper:
- NAUT: Number of authors
- NCOAUT: Number of coauthors (across all authors)
- MAXCOAUT: Maximum coauthors of any author
- NPUB: Number of previous publications (across all authors)
- MAXPUB: Maximum publications of any author

These models capture author influence on citation probability.
"""

import numpy as np
from enum import Enum
from .multilayer_network import MultilayerNetwork


class SocialConstituentType(Enum):
    """Enumeration of available social constituent model types."""
    NAUT = "NAUT"
    NCOAUT = "NCOAUT"
    MAXCOAUT = "MAXCOAUT"
    NPUB = "NPUB"
    MAXPUB = "MAXPUB"


class SocialConstituents:
    """Collection of social constituent models η_j."""

    @staticmethod
    def naut(network: MultilayerNetwork, pub_id: int, current_time: float) -> float:
        """
        Number of authors: η_j = k^ap_j(t)

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t

        Returns:
            Number of authors for this publication
        """
        authors = network.get_authors(pub_id)
        return float(len(authors))

    @staticmethod
    def ncoaut(network: MultilayerNetwork, pub_id: int, current_time: float) -> float:
        """
        Number of coauthors: η_j = |{v | ∃λ^ap_{3,jv}(t)}|

        Total number of unique coauthors across all authors of publication j.

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t

        Returns:
            Total number of unique coauthors
        """
        authors = network.get_authors(pub_id)
        if len(authors) == 0:
            return 0.0
        
        # Vectorized coauthor collection using set union
        all_coauthors = set()
        coauthor_sets = [set(network.get_coauthors(author_id, current_time)) for author_id in authors]
        for coauthors in coauthor_sets:
            all_coauthors.update(coauthors)

        return float(len(all_coauthors))

    @staticmethod
    def maxcoaut(network: MultilayerNetwork, pub_id: int, current_time: float) -> float:
        """
        Maximum coauthors: η_j = max_{q∈authors(j)} |{v | ∃λ^ap_{2,qv}(t)}|

        Maximum number of coauthors of any single author of publication j.

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t

        Returns:
            Maximum number of coauthors among all authors
        """
        authors = network.get_authors(pub_id)
        if len(authors) == 0:
            return 0.0

        # Vectorized max coauthors calculation
        coauthor_counts = np.array([len(network.get_coauthors(author_id, current_time)) 
                                   for author_id in authors])
        return float(np.max(coauthor_counts))

    @staticmethod
    def npub(network: MultilayerNetwork, pub_id: int, current_time: float) -> float:
        """
        Number of publications: η_j = |{v | ∃λ^ap_{2,jv}(t)}|

        Total number of unique previous publications across all authors of publication j.

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t

        Returns:
            Total number of unique previous publications
        """
        authors = network.get_authors(pub_id)
        if len(authors) == 0:
            return 0.0
        
        # Vectorized previous publications collection using set union
        all_prev_pubs = set()
        pub_sets = [set(network.get_publications_by_author(author_id, current_time)) for author_id in authors]
        for pubs in pub_sets:
            all_prev_pubs.update(pubs)

        return float(len(all_prev_pubs))

    @staticmethod
    def maxpub(network: MultilayerNetwork, pub_id: int, current_time: float) -> float:
        """
        Maximum publications: η_j = max_{q∈authors(j)} k^ap_q(t)

        Maximum number of previous publications of any single author of publication j.

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t

        Returns:
            Maximum number of previous publications among all authors
        """
        authors = network.get_authors(pub_id)
        if len(authors) == 0:
            return 0.0

        # Vectorized max publications calculation
        pub_counts = np.array([len(network.get_publications_by_author(author_id, current_time)) 
                              for author_id in authors])
        return float(np.max(pub_counts))

    @staticmethod
    def compute_social_constituent(network: MultilayerNetwork, pub_id: int,
                                 current_time: float, model_type: str) -> float:
        """
        Compute social constituent η_j for a given model type.

        Args:
            network: MultilayerNetwork instance
            pub_id: Publication ID
            current_time: Current time t
            model_type: One of "NAUT", "NCOAUT", "MAXCOAUT", "NPUB", "MAXPUB"

        Returns:
            Social constituent value η_j

        Raises:
            ValueError: If model_type is not recognized
        """
        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_enum = SocialConstituentType(model_type)
            except ValueError:
                available_types = [model.value for model in SocialConstituentType]
                raise ValueError(
                    f"Unknown social constituent type: {model_type}. "
                    f"Must be one of {available_types}"
                )
        else:
            model_enum = model_type

        # Use match-case for dispatch
        match model_enum:
            case SocialConstituentType.NAUT:
                return SocialConstituents.naut(network, pub_id, current_time)
            case SocialConstituentType.NCOAUT:
                return SocialConstituents.ncoaut(network, pub_id, current_time)
            case SocialConstituentType.MAXCOAUT:
                return SocialConstituents.maxcoaut(network, pub_id, current_time)
            case SocialConstituentType.NPUB:
                return SocialConstituents.npub(network, pub_id, current_time)
            case SocialConstituentType.MAXPUB:
                return SocialConstituents.maxpub(network, pub_id, current_time)
            case _:
                available_types = [model.value for model in SocialConstituentType]
                raise ValueError(
                    f"Unknown social constituent type: {model_enum}. "
                    f"Must be one of {available_types}"
                )

    @staticmethod
    def compute_all_social_constituents(network: MultilayerNetwork,
                                      current_time: float, model_type: str) -> np.ndarray:
        """
        Compute social constituents for all publications at current time.

        Args:
            network: MultilayerNetwork instance
            current_time: Current time t
            model_type: Social constituent model type

        Returns:
            numpy array of social constituent values
        """
        available_pubs = network.get_publications_at_time(current_time)
        n_pubs = len(available_pubs)

        if n_pubs == 0:
            return np.array([])

        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_enum = SocialConstituentType(model_type)
            except ValueError:
                available_types = [model.value for model in SocialConstituentType]
                raise ValueError(f"Unknown social constituent type: {model_type}. Must be one of {available_types}")
        else:
            model_enum = model_type

        # Vectorized computation based on model type
        match model_enum:
            case SocialConstituentType.NAUT:
                # Vectorized: Number of authors for each publication
                all_authors = network.get_all_authors(available_pubs)
                author_counts = np.array([len(authors) for authors in all_authors], dtype=np.float64)
                return author_counts
            
            case SocialConstituentType.NCOAUT:
                # Vectorized: Total unique coauthors across all authors of each publication
                def get_unique_coauthors(pub_id):
                    authors = network.get_authors(pub_id)
                    if len(authors) == 0:
                        return 0
                    # Get all coauthors for all authors of this publication
                    all_coauthors = set()
                    for author_id in authors:
                        coauthors = network.get_coauthors(author_id, current_time)
                        all_coauthors.update(coauthors)
                    return len(all_coauthors)
                
                constituents = np.array([get_unique_coauthors(pub_id) for pub_id in available_pubs],
                                      dtype=np.float64)
                return constituents
            
            case SocialConstituentType.MAXCOAUT:
                # Vectorized: Maximum coauthors of any author in each publication
                def get_max_coauthors(pub_id):
                    authors = network.get_authors(pub_id)
                    if len(authors) == 0:
                        return 0
                    # Get coauthor counts for all authors and return max
                    coauthor_counts = np.array([len(network.get_coauthors(author_id, current_time)) 
                                              for author_id in authors])
                    return np.max(coauthor_counts) if len(coauthor_counts) > 0 else 0
                
                constituents = np.array([get_max_coauthors(pub_id) for pub_id in available_pubs],
                                      dtype=np.float64)
                return constituents
            
            case SocialConstituentType.NPUB:
                # Vectorized: Total unique previous publications across all authors
                def get_unique_prev_pubs(pub_id):
                    authors = network.get_authors(pub_id)
                    if len(authors) == 0:
                        return 0
                    # Get all previous publications for all authors of this publication
                    all_prev_pubs = set()
                    for author_id in authors:
                        prev_pubs = network.get_publications_by_author(author_id, current_time)
                        all_prev_pubs.update(prev_pubs)
                    return len(all_prev_pubs)
                
                constituents = np.array([get_unique_prev_pubs(pub_id) for pub_id in available_pubs],
                                      dtype=np.float64)
                return constituents
            
            case SocialConstituentType.MAXPUB:
                # Vectorized: Maximum previous publications of any author
                def get_max_prev_pubs(pub_id):
                    authors = network.get_authors(pub_id)
                    if len(authors) == 0:
                        return 0
                    # Get publication counts for all authors and return max
                    pub_counts = np.array([len(network.get_publications_by_author(author_id, current_time)) 
                                         for author_id in authors])
                    return np.max(pub_counts) if len(pub_counts) > 0 else 0
                
                constituents = np.array([get_max_prev_pubs(pub_id) for pub_id in available_pubs],
                                      dtype=np.float64)
                return constituents
            
            case _:
                available_types = [model.value for model in SocialConstituentType]
                raise ValueError(f"Unknown social constituent type: {model_enum}. Must be one of {available_types}")

    @staticmethod
    def get_available_models() -> np.ndarray:
        """Get array of available social constituent model types."""
        return np.array([model.value for model in SocialConstituentType])