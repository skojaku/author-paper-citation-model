# %%
import sys
from typing import Dict
import numpy as np
import pandas as pd
from multilayer_citation_model import (
    MultilayerNetwork,
    ModelConfig,
    PaperSyntheticGenerator,
    infer_parameters,
    compare_models,
    OptimizationResult,
)

"""Set up test fixtures."""
# Set random seed for reproducibility
np.random.seed(42)

# Common test parameters
n_pubs = 2000
n_authors = 2000
true_alpha = 0.7
true_delta = 1.2

generator = PaperSyntheticGenerator(n_pubs=n_pubs, n_authors=n_authors, seed=42)
network, citations = generator.generate_two_layer_network(true_alpha, true_delta)
config = ModelConfig("PA", "NAUT", "ADDITIVE")

n_repeats = 5
sample_sizes = [10, 50, 100, 200, 500]
results = []

for repeat in range(n_repeats):
    for sample_size in sample_sizes:
        print(f"Testing sample size: {sample_size}")

        result = infer_parameters(
            network, citations, sample_size, config, seed=np.random.randint(1000)
        )

        if result["success"]:
            estimated_alpha = result["alpha"]
            estimated_delta = result["delta"]

            results.append(
                {
                    "sample_size": sample_size,
                    "alpha_est": estimated_alpha,
                    "delta_est": estimated_delta,
                    "log_likelihood": result["log_likelihood"],
                }
            )

            print(f"  α: {estimated_alpha:.3f} (true: {true_alpha})")
            print(f"  δ: {estimated_delta:.3f} (true: {true_delta})")
            print(f"  log_likelihood: {result['log_likelihood']}")

result_table = pd.DataFrame(results)
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(data=result_table, x="sample_size", y="alpha_est", marker="o", ax=ax[0])
sns.lineplot(data=result_table, x="sample_size", y="delta_est", marker="o", ax=ax[1])

ax[0].axhline(true_alpha, color="red", linestyle="--")
ax[1].axhline(true_delta, color="red", linestyle="--")

ax[0].set_ylabel("α")
ax[1].set_ylabel("δ")
ax[0].set_xlabel("Sample size")
ax[1].set_xlabel("Sample size")

sns.despine()
fig.savefig("fig4.pdf", bbox_inches="tight")

# %%
