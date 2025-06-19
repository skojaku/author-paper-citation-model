"""
Synthetic Network Generation

Implementation of synthetic network generation following the methodology
from Section IV of the paper:
"Multilayer network approach to modeling authorship influence on citation dynamics in physics journals"
Physical Review E 102, 032303 (2020)

Main components:
- PaperSyntheticGenerator: Generates two-layer networks with known parameters
- paper_time_step_sampling: Implements the paper's time-step sampling methodology
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple
from .multilayer_network import MultilayerNetwork




class PaperSyntheticGenerator:
    """
    Synthetic network generator following Section IV of the paper.

    Implements the two-layer network with preferential attachment mechanism
    from Equation 11 in the paper.
    """

    def __init__(self, n_pubs: int = 2000, n_authors: int = 2000, seed: int = 42):
        """
        Initialize generator with paper specifications.

        From Section IV: "we choose V_p(T) = V_a(T) = 2000 nodes in each of the two layers.
        Adding one node at a time to layer p, this implies a total growth period of T = 2000."
        """
        self.n_pubs = n_pubs
        self.n_authors = n_authors
        np.random.seed(seed)

    def generate_two_layer_network(
        self, alpha: float = 0.7, delta: float = 1.2
    ) -> Tuple[MultilayerNetwork, List[Tuple[int, int, float]]]:
        """
        Generate synthetic two-layer network following paper's methodology.

        From Section IV: "When adding an edge from the newly added node i ∈ V_p,
        we choose the existing node j ∈ V_p with probability:

        π^pp,true_ij[G(t); α, δ] = α * [k^pp,in_j(t) + δ] / [Σ_l k^pp,in_l(t) + δ]
                                  + (1-α) * [k^ap_j] / [Σ_l k^ap_l]"

        Args:
            alpha: Mixture weight (0.7 in paper)
            delta: Offset parameter (1.2 in paper)

        Returns:
            (MultilayerNetwork, citation_list)
        """
        print(f"Generating two-layer network with α={alpha}, δ={delta}")

        # Step 1: Create authorship connections (V_a fixed, connected to V_p)
        authorship_rows, authorship_cols = [], []

        # Each publication connects to 1-5 authors uniformly as in paper
        for pub_id in range(self.n_pubs):
            n_authors_for_pub = np.random.randint(1, 6)  # 1 to 5 authors
            selected_authors = np.random.choice(
                self.n_authors, size=n_authors_for_pub, replace=False
            )
            for author_id in selected_authors:
                authorship_rows.append(author_id)
                authorship_cols.append(pub_id)

        authorship_matrix = sparse.csr_matrix(
            (np.ones(len(authorship_rows)), (authorship_rows, authorship_cols)),
            shape=(self.n_authors, self.n_pubs),
        )

        # Step 2: Generate citations using two-layer preferential attachment
        citation_rows, citation_cols = [], []
        citations_list = []

        # Track state variables
        in_degrees = np.zeros(self.n_pubs, dtype=int)  # k^pp,in_j(t)
        author_connections = authorship_matrix.sum(axis=0).A1  # k^ap_j

        # Add nodes one at a time with out-degree 1-5 (as in paper)
        for citing_pub in range(1, self.n_pubs):
            current_time = float(citing_pub)

            # Available publications to cite (published before current time)
            available_pubs = np.arange(citing_pub)
            if len(available_pubs) == 0:
                continue

            # Number of outgoing citations
            # Paper says "out-degree randomly drawn between one and five"
            # But we need to ensure balance - use adaptive approach
            max_citations = 5
            n_citations = np.random.randint(1, max_citations + 1)

            # Mechanism 1: Citation constituent (preferential attachment within layer)
            citation_scores = in_degrees[available_pubs].astype(float) + delta
            citation_probs = citation_scores / np.sum(citation_scores)

            # Mechanism 2: Social constituent (connections across layers)
            social_scores = author_connections[available_pubs]
            if np.sum(social_scores) > 0:
                social_probs = social_scores / np.sum(social_scores)
            else:
                social_probs = np.ones(len(available_pubs)) / len(available_pubs)

            # Combined probability (Equation 11)
            base_probs = alpha * citation_probs + (1 - alpha) * social_probs

            # Sample citations without replacement (handle case where n_citations > available_pubs)
            n_citations = min(n_citations, len(available_pubs))
            cited_pubs = np.random.choice(
                available_pubs, size=n_citations, replace=False, p=base_probs
            )

            # Add citations to network
            for cited_pub in cited_pubs:
                citation_rows.append(citing_pub)
                citation_cols.append(cited_pub)
                citations_list.append((citing_pub, cited_pub, current_time))

                # Update in-degree for cited publication
                in_degrees[cited_pub] += 1

        # Build sparse citation matrix
        citation_matrix = sparse.csr_matrix(
            (np.ones(len(citation_rows)), (citation_rows, citation_cols)),
            shape=(self.n_pubs, self.n_pubs),
        )

        network = MultilayerNetwork(citation_matrix, authorship_matrix)
        print(f"Generated network with {len(citations_list)} citations")
        return network, citations_list