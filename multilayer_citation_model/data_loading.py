"""
Data loading and network construction utilities for the citation inference model.

Functions for building MultilayerNetwork from pandas DataFrames and handling
temporal data splitting.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy import sparse

from .multilayer_network import MultilayerNetwork


def load_network_from_dataframe(
    publications_df: pd.DataFrame,
    citations_df: pd.DataFrame,
    pub_id_col: str = "publication_id",
    time_col: str = "timestamp",
    authors_col: str = "authors",
    citing_col: str = "citing_pub",
    cited_col: str = "cited_pub",
) -> MultilayerNetwork:
    """
    Load MultilayerNetwork from pandas DataFrames.

    Args:
        publications_df: DataFrame with publication metadata
        citations_df: DataFrame with citation edges
        pub_id_col: Column name for publication IDs
        time_col: Column name for timestamps
        authors_col: Column name for authors (list or comma-separated string)
        citing_col: Column name for citing publication
        cited_col: Column name for cited publication

    Returns:
        MultilayerNetwork instance
    """
    # Map publication IDs to indices (0...N-1)
    pub_ids = publications_df[pub_id_col].astype(int).values
    pub_id_to_idx = {pid: idx for idx, pid in enumerate(pub_ids)}
    idx_to_pub_id = {idx: pid for idx, pid in enumerate(pub_ids)}
    n_pubs = len(pub_ids)

    # Build author list and mapping
    def process_authors(authors_data):
        if isinstance(authors_data, str):
            return [int(a.strip()) for a in authors_data.split(",") if a.strip()]
        elif isinstance(authors_data, list):
            return [int(a) for a in authors_data]
        else:
            return [int(authors_data)]

    authors_set = set()
    authors_per_pub = []
    for authors_data in publications_df[authors_col]:
        authors = process_authors(authors_data)
        authors_per_pub.append(authors)
        authors_set.update(authors)
    author_ids = sorted(list(authors_set))
    author_id_to_idx = {aid: idx for idx, aid in enumerate(author_ids)}
    n_authors = len(author_ids)

    # Build authorship matrix (n_authors x n_pubs)
    authorship_rows = []
    authorship_cols = []
    for pub_idx, authors in enumerate(authors_per_pub):
        for author in authors:
            author_idx = author_id_to_idx[author]
            authorship_rows.append(author_idx)
            authorship_cols.append(pub_idx)
    authorship_data = np.ones(len(authorship_rows), dtype=np.int8)
    authorship_matrix = sparse.csr_matrix(
        (authorship_data, (authorship_rows, authorship_cols)), shape=(n_authors, n_pubs)
    )

    # Build citation matrix (n_pubs x n_pubs)
    citing_ids = citations_df[citing_col].astype(int).values
    cited_ids = citations_df[cited_col].astype(int).values
    citing_idx = [
        pub_id_to_idx[cid]
        for cid in citing_ids
        if cid in pub_id_to_idx
        and cited_ids[list(citing_ids).index(cid)] in pub_id_to_idx
    ]
    cited_idx = [
        pub_id_to_idx[cid]
        for cid in cited_ids
        if cid in pub_id_to_idx
        and citing_ids[list(cited_ids).index(cid)] in pub_id_to_idx
    ]
    # To ensure both lists are aligned, filter together
    # Also filter out invalid citations where citing_pub <= cited_pub (temporal constraint)
    citation_pairs = [
        (pub_id_to_idx[citing], pub_id_to_idx[cited])
        for citing, cited in zip(citing_ids, cited_ids)
        if citing in pub_id_to_idx and cited in pub_id_to_idx and citing > cited
    ]
    if citation_pairs:
        citation_rows, citation_cols = zip(*citation_pairs)
    else:
        citation_rows, citation_cols = [], []
    citation_data = np.ones(len(citation_rows), dtype=np.int8)
    citation_matrix = sparse.csr_matrix(
        (citation_data, (citation_rows, citation_cols)), shape=(n_pubs, n_pubs)
    )

    # Construct MultilayerNetwork
    network = MultilayerNetwork(
        citation_matrix=citation_matrix, authorship_matrix=authorship_matrix
    )
    return network


def split_temporal_data(
    network: MultilayerNetwork, split_time: float
) -> Tuple[MultilayerNetwork, List[Tuple[int, int, float]]]:
    """
    Split network into training and test sets based on time.

    Args:
        network: Full MultilayerNetwork
        split_time: Time threshold for train/test split

    Returns:
        (training_network, test_citations)
    """
    # Create training network up to split_time
    train_network = network.get_network_at_time(split_time)

    # Extract test citations after split_time
    # Since pub_time = publication_id, we need to extract citation edges and their times
    citation_matrix = network.citation_matrix
    n_pubs = citation_matrix.shape[0]
    test_citations = []

    for citing_idx in range(n_pubs):
        citing_time = network.get_publication_time(
            citing_idx
        )  # This is just float(citing_idx)
        if citing_time > split_time:
            cited_indices = citation_matrix[citing_idx, :].nonzero()[1]
            for cited_idx in cited_indices:
                test_citations.append((citing_idx, cited_idx, citing_time))
    return train_network, test_citations


# Alias for backward compatibility
build_network_from_dataframes = load_network_from_dataframe