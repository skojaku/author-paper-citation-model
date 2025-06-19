"""
MultilayerNetwork: Efficient sparse representation of citation and authorship networks

This module implements the core data structure for the multilayer citation model,
using scipy.sparse matrices for memory efficiency and fast operations.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple
from collections import defaultdict


class MultilayerNetwork:
    """
    Multilayer network representation with sparse matrices for efficiency.

    Manages:
    - V_p(t): Publication nodes over time
    - V_a(t): Author nodes over time
    - E_pp(t): Citation edges between publications (sparse CSR matrix)
    - E_ap(t): Authorship edges between authors and publications (sparse CSR matrix)
    """

    def __init__(
        self,
        citation_matrix: sparse.csr_matrix,
        authorship_matrix: sparse.csr_matrix,
        # pub_ids: np.ndarray,
    ):
        """
        Initialize MultilayerNetwork with pre-built sparse matrices.

        Args:
            citation_matrix: CSR matrix of shape (n_pubs, n_pubs) representing citations
            authorship_matrix: CSR matrix of shape (n_authors, n_pubs) representing authorship
        """
        # Sparse matrices
        self.citation_matrix = citation_matrix.tocsr()  # E_pp: pubs x pubs
        self.citation_matrix_t = (
            self.citation_matrix.T.tocsr()
        )  # E_pp^T: pubs x pubs (transposed for fast column ops)
        self.authorship_matrix = authorship_matrix.tocsr()  # E_ap: authors x pubs
        self.authorship_matrix_t = (
            self.authorship_matrix.T.tocsr()
        )  # E_ap^T: pubs x authors

        # Store dimensions for reference
        self.n_pubs = self.citation_matrix.shape[0]
        self.n_authors = self.authorship_matrix.shape[0]

    def get_publications_at_time(self, time: float) -> np.ndarray:
        """Get all publications at or before given time."""
        # Since pub_time = publication_id, we can directly filter
        if time >= self.n_pubs:
            return np.arange(self.n_pubs, dtype=np.int32)
        return np.arange(int(time) + 1, dtype=np.int32)

    def get_in_degree(self, pub_id: int, time: Optional[float] = None) -> int:
        """Get in-degree (citation count) of a publication at given time."""
        if self.citation_matrix is None:
            return 0

        # If time is specified, only count citations from papers published before time
        if time is not None:
            # Since pub_time = publication_id, we can directly filter citing papers
            time_idx = int(np.minimum(time + 1, self.citation_matrix.shape[0]))
            return int(self.citation_matrix_t[pub_id, :time_idx].sum())
        return int(self.citation_matrix_t[pub_id, :].sum())

    def get_authors(self, pub_id: int) -> np.ndarray:
        """Get authors of a publication."""
        # Find authors using transposed matrix for fast operation
        author_indices = self.authorship_matrix_t[pub_id, :].nonzero()[1]
        return author_indices

    def get_publications_by_author(
        self, author_id: int, before_time: Optional[float] = None
    ) -> np.ndarray:
        """Get all publications by an author, optionally before a given time."""

        # Find publications (columns) that this author (row) has edges to
        pubs = self.authorship_matrix[author_id, :].nonzero()[1]

        # Filter by time if specified
        if before_time is not None:
            # Since pub_time = publication_id, we can directly filter
            pubs = pubs[pubs < before_time]

        return pubs

    def get_coauthors(
        self, author_id: int, before_time: Optional[float] = None
    ) -> np.ndarray:
        """Get all coauthors of an author, optionally before a given time."""
        # Get all publications by this author
        author_pubs = self.get_publications_by_author(author_id, before_time)

        if len(author_pubs) == 0:
            return np.array([], dtype=np.int32)

        # Since pub_ids are ordered indices, author_pubs directly map to matrix indices
        # Get all coauthors from these publications using transposed matrix for fast operation
        coauthors = np.unique(self.authorship_matrix_t[author_pubs, :].nonzero()[1])

        # Remove the author themselves
        coauthors = coauthors[coauthors != author_id]
        return coauthors

    def get_publication_time(self, pub_id: int) -> float:
        """Get publication timestamp."""
        # Since pub_time = publication_id and pub_ids are ordered indices
        return float(pub_id)

    def get_publication_times(self, pub_ids: np.ndarray) -> np.ndarray:
        """Get publication timestamps for multiple publications (vectorized)."""
        # Since pub_time = publication_id, we can directly return pub_ids
        return pub_ids.astype(np.float64)

    def get_in_degrees(
        self, pub_ids: np.ndarray, time: Optional[float] = None
    ) -> np.ndarray:
        """Get in-degrees (citation counts) for multiple publications (vectorized)."""
        if time is not None:
            # Since pub_time = publication_id, we can directly filter citing papers
            time_idx = int(np.minimum(time + 1, self.citation_matrix.shape[0]))
            degrees = self.citation_matrix_t[pub_ids, :time_idx].sum(axis=1)
        else:
            degrees = self.citation_matrix_t[pub_ids, :].sum(axis=1)

        return np.array(degrees).flatten().astype(np.int32)

    def get_all_authors(self, pub_ids: np.ndarray) -> list:
        """Get authors for multiple publications (vectorized)."""
        authors_list = []
        for pub_id in pub_ids:
            authors_list.append(self.get_authors(pub_id))
        return authors_list

    def get_network_at_time(self, time: float) -> "MultilayerNetwork":
        """Create a snapshot of the network at a given time."""
        # Filter citation matrix to only include valid publications
        time_idx = int(np.minimum(time + 1, self.citation_matrix.shape[0]))
        valid_citation_matrix = self.citation_matrix[:time_idx, :time_idx]

        # Get all authors that appear in valid publications
        valid_authorship_matrix = self.authorship_matrix[:, :time_idx]

        return MultilayerNetwork(
            citation_matrix=valid_citation_matrix,
            authorship_matrix=valid_authorship_matrix,
        )

    def __repr__(self) -> str:
        return f"MultilayerNetwork(pubs={self.n_pubs}, authors={self.n_authors})"
