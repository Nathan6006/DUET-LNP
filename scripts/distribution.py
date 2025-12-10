import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/all_data.csv")
tox = df["quantified_toxicity"].dropna()

# Custom bins
bins = [0.0, 0.8, 0.9, 1.0]

# Histogram
counts, edges = np.histogram(tox, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2

# Print counts
for i in range(len(counts)):
    print(f"{edges[i]} to {edges[i+1]}: {counts[i]}")

# Plot
plt.figure(figsize=(8, 5))
plt.bar(centers, counts, width=0.08)

plt.xlabel("Quantified Toxicity")
plt.ylabel("Count")
plt.title("Distribution of Quantified Toxicity")
plt.xticks(bins)
plt.tight_layout()
plt.show()