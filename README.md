# Multilayer Citation Model

A Python package for modeling and predicting citations in multilayer networks, combining citation dynamics with social network effects.

> PHYSICAL REVIEW E 102, 032303 (2020)
> Multilayer network approach to modeling authorship influence on
> citation dynamics in physics journals
> Vahan Nanumyan, Christoph Gote, and Frank Schweitzer


Implemented by [Sadamori Kojaku](https://github.com/skojaku).

Disclaimer: While I tried to implement the model as described in the paper, there could be some bugs and inconsistencies. Please report if you find any.

## 🚀 Quick Start

```python
from multilayer_citation_model import MultilayerNetwork, CitationInference, ModelConfig
from multilayer_citation_model.utils import generate_synthetic_network

# Create a synthetic network
network = generate_synthetic_network(n_pubs=100, n_authors=50)

# Configure model (PA-RD with social effects)
config = ModelConfig("PA-RD", "NCOAUT", "ADDITIVE")

# Create inference model
theta = [0.5, 0.0, 1.0, 1.0, 100.0]  # [α, β, γ, δ, τ]
inference = CitationInference(network, theta, config)

# Predict top 10 most likely citations at time 50
predictions = inference.predict_citations(-1, 50.0, top_k=10)
print("Top 10 most likely papers to be cited:")
for i, (pub_id, prob) in enumerate(predictions, 1):
    print(f"{i:2d}. Paper {pub_id}: {prob:.6f}")
```

## 📦 Installation

### From Source

```bash
# Install in development mode
pip install -e .
```

## ⚠️ Key Assumptions

This package makes several important assumptions for performance optimization:

1. **Chronological Publication IDs**: Publication IDs must correspond to chronological time ordering
   - If `i < j`, then publication `i` appears before publication `j`
   - The code assumes `publication_time = publication_id`
   - Publication IDs should be sequential starting from 0: 0, 1, 2, 3, ...

2. **Citation Direction**: The model only considers citations from later to earlier publications
   - Citations where `citing_pub > cited_pub` are valid and used in the model
   - Citations where `citing_pub <= cited_pub` are automatically filtered out
   - You can provide edge tables with citations in both directions - invalid ones will be ignored

3. **Data Preprocessing**:
   - Publication IDs must be preprocessed to be chronologically ordered
   - Citation edge tables can be provided as-is (invalid citations will be automatically filtered)

## 🏗️ Core Components

### 1. MultilayerNetwork
Core data structure for citation and authorship networks using sparse matrices for efficiency.

```python
from multilayer_citation_model import MultilayerNetwork
import numpy as np
from scipy import sparse

# Create matrices
n_pubs, n_authors = 100, 50
citation_matrix = sparse.random(n_pubs, n_pubs, density=0.05)
authorship_matrix = sparse.random(n_authors, n_pubs, density=0.1)

# Initialize network
network = MultilayerNetwork(citation_matrix, authorship_matrix)
```

### 2. Citation Models
Various citation constituent models implementing different attachment mechanisms:

- **UNIF**: Uniform attachment (baseline)
- **PA**: Preferential attachment
- **PA-RD**: Preferential attachment with relevance decay
- **PA-NL-RD**: Nonlinear preferential attachment with relevance decay

### 3. Social Constituents
Social network features capturing author influence:

- **NAUT**: Number of authors
- **NCOAUT**: Number of unique coauthors
- **MAXCOAUT**: Maximum coauthors of any author
- **NPUB**: Number of previous publications
- **MAXPUB**: Maximum publications of any author

### 4. Model Configuration
Combine citation and social models with different coupling strategies:

```python
from multilayer_citation_model import ModelConfig

# Citation-only model
config1 = ModelConfig("PA-RD", None, "NONE")

# Social + Citation with additive coupling
config2 = ModelConfig("PA-RD", "NCOAUT", "ADDITIVE")

# Social + Citation with multiplicative coupling
config3 = ModelConfig("PA-RD", "NCOAUT", "MULTIPLICATIVE")
```

## 🔧 Advanced Usage

### Parameter Optimization

```python
from multilayer_citation_model.utils import optimize_parameters, split_temporal_data

# Split network into train/test
train_network, test_citations = split_temporal_data(network, split_time=75.0)

# Extract training citations
train_citations = []
for citing_idx in range(train_network.n_pubs):
    citing_time = train_network.get_publication_time(citing_idx)
    cited_indices = train_network.citation_matrix[citing_idx, :].nonzero()[1]
    for cited_idx in cited_indices:
        train_citations.append((citing_idx, cited_idx, citing_time))

# Optimize parameters (uses fast pre-computation by default)
config = ModelConfig("PA-RD", "NCOAUT", "ADDITIVE")
result = optimize_parameters(train_network, train_citations, config)

print(f"Optimal parameters: {result.optimal_params}")
print(f"Log-likelihood: {result.log_likelihood}")
print(f"Converged in {result.iterations} iterations")
```

### Model Comparison

```python
from multilayer_citation_model.utils import compare_models

# Define multiple model configurations
models = [
    ("UNIF", ModelConfig("UNIF", None, "NONE")),
    ("PA", ModelConfig("PA", None, "NONE")),
    ("PA-RD", ModelConfig("PA-RD", None, "NONE")),
    ("PA-RD + Social", ModelConfig("PA-RD", "NCOAUT", "ADDITIVE")),
]

# Optimize each model and compare
results = []
model_names = []

for name, config in models:
    result = optimize_parameters(train_network, train_citations, config)
    if result.success:
        results.append(result)
        model_names.append(name)

# Compare using AIC
comparison_df = compare_models(results, model_names)
print(comparison_df[["model", "log_likelihood", "aic", "delta_aic"]])
```

**⚠️ Data Requirements**:
- Publication IDs must start from 0 and be sequential: 0, 1, 2, 3, ...
- Publication IDs must correspond to chronological order
- Citation edge table can contain citations in both directions (the code will filter to keep only valid ones)
- Only citations where `citing_pub > cited_pub` will be used in the model

## 🎯 Examples

Run the included examples:

```python
from multilayer_citation_model import examples

# Basic citation inference
examples.basic_example()

# Model comparison
examples.model_comparison_example()
```

Or run from command line:
```bash
multilayer-citation-model  # Runs basic example
```

## 📊 Performance Optimizations

The package includes several performance optimizations:

### Fast Parameter Optimization
- **Pre-computation**: Network statistics are computed once and cached
- **Vectorized operations**: Efficient numpy/scipy operations
- **Sparse matrices**: Memory-efficient storage for large networks

```python
# Fast optimization (default)
result = optimize_parameters(network, citations, config, use_fast=True)

# Original slower version (for comparison)
result = optimize_parameters(network, citations, config, use_fast=False)
```

## 📚 API Reference

### Core Classes

#### MultilayerNetwork
```python
MultilayerNetwork(citation_matrix, authorship_matrix)
```

**Key Methods:**
- `get_publications_at_time(time)`: Get publications available at time
- `get_in_degree(pub_id, time=None)`: Get citation count
- `get_authors(pub_id)`: Get authors of publication
- `get_coauthors(author_id, before_time=None)`: Get coauthors
- `get_network_at_time(time)`: Create temporal subnetwork

#### CitationInference
```python
CitationInference(network, theta, config)
```

**Key Methods:**
- `citation_probability(citing_pub, cited_pub, time)`: Calculate probability
- `predict_citations(citing_pub, time, top_k=None)`: Rank predictions
- `sample_citations(citing_pub, time, n_samples, seed=None)`: Sample citations

#### ModelConfig
```python
ModelConfig(citation_type, social_type=None, coupling_type="NONE")
```

**Parameters:**
- `citation_type`: "UNIF", "PA", "PA-RD", "PA-NL-RD"
- `social_type`: None, "NAUT", "NCOAUT", "MAXCOAUT", "NPUB", "MAXPUB"
- `coupling_type`: "NONE", "ADDITIVE", "MULTIPLICATIVE"

### Utility Functions

#### Optimization
- `optimize_parameters(network, citations, config, ...)`: Optimize model parameters
- `optimize_parameters_fast(...)`: Fast version with pre-computation
- `precompute_network_stats(...)`: Pre-compute network statistics

#### Data Processing
- `generate_synthetic_network(n_pubs, n_authors, ...)`: Generate test data
- `build_network_from_dataframes(...)`: Build from pandas DataFrames
- `split_temporal_data(network, split_time)`: Train/test split

#### Model Evaluation
- `compare_models(results, model_names)`: Compare multiple models
- `calculate_aic(log_likelihood, num_params)`: Akaike Information Criterion
- `calculate_bic(log_likelihood, num_params, num_obs)`: Bayesian Information Criterion

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/skojaku/Legal-Citations.git
cd Legal-Citations/libs/authorship_citation_model

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=multilayer_citation_model --cov-report=html
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.