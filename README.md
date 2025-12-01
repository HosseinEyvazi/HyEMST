# HyEMST: Hybrid Density-Distance Clustering with Maximum Spanning Trees

This repository contains the official implementation for the paper: **"HyEMST: A Hybrid Density-Distance Clustering Algorithm with Maximum Spanning Trees for Non-Convex Geometries"**.

HyEMST is a novel clustering algorithm designed to overcome the limitations of traditional methods like K-Means and DBSCAN. It effectively identifies clusters of arbitrary shapes and varying densities by unifying geometric distance and volumetric density within a principled five-phase framework.

  <!-- Replace with a URL to your pipeline figure -->

## Key Features

- **Handles Non-Convex Geometries**: Successfully clusters datasets with arbitrary shapes (e.g., crescents, rings) where K-Means fails.
- **Robust to Varying Densities**: Unlike DBSCAN, HyEMST can simultaneously identify both dense and sparse clusters.
- **Parameter-Free Density Estimation**: Uses Minimum Volume Enclosing Ellipsoids (MVEE) to estimate local density without requiring sensitive bandwidth parameters.
- **Adaptive Merging Strategy**: Employs a density-aware threshold to intelligently merge micro-clusters, respecting local data characteristics.
- **Bayesian Hyperparameter Optimization**: Integrates **Optuna** to systematically find the optimal set of hyperparameters for any given dataset.

## Methodology Overview

The HyEMST algorithm operates through a five-phase pipeline:

1.  **Geometric Decomposition**: The input data is first over-clustered into a set of small, convex micro-clusters using K-Means with random initialization. This phase breaks down complex non-convex structures into manageable components.
2.  **Volumetric Density Estimation**: The local density of each micro-cluster is estimated using the ratio of its point count to the volume of its Minimum Volume Enclosing Ellipsoid (MVEE). This is a parameter-free and shape-adaptive approach.
3.  **Hybrid Kernel Construction**: A hybrid affinity matrix is constructed by taking the weighted geometric mean of a distance-based kernel and a density-based kernel. The trade-off is controlled by the parameter `λ`.
4.  **Topological Structure Discovery**: A Maximum Spanning Tree (MST) is built on the graph of micro-clusters using the hybrid affinity matrix. This reduces the graph's complexity from O(K²) to O(K) while preserving the complete single-linkage hierarchy.
5.  **Adaptive Density-Aware Merging**: The final clusters are formed by cutting edges in the MST. The cut threshold is adapted based on the local densities of the connected micro-clusters, allowing sparse regions to merge more readily than dense ones.

## Repository Structure

This repository provides Jupyter notebooks to replicate the experiments on five benchmark datasets. Each notebook is a self-contained script that runs the full Bayesian hyperparameter optimization, generates final clustering results, and produces all evaluation metrics and visualizations.

-   `HyEMST - JAIN.ipynb`: Runs HyEMST on the non-convex Jain dataset.
-   `HyEMST - AGGREGATION.ipynb`: Runs HyEMST on the Aggregation dataset, which contains 7 arbitrary-shaped clusters.
-   `HyEMST - R15.ipynb`: Runs HyEMST on the R15 dataset (15 Gaussian clusters).
-   `HyEMST - D31.ipynb`: Runs HyEMST on the D31 dataset (31 Gaussian clusters).
-   `HyEMST - NOISY CIRCLES.ipynb`: Runs HyEMST on two concentric circles with noise.
-   `requirements.txt`: Contains all necessary Python packages.

## Installation & Usage

### 1. Prerequisites

-   Python 3.8+
-   `pip` package manager

### 2. Installation

Clone the repository and install the required packages.

git clone https://github.com/HosseinEyvazi/HyEMST.git
cd HyEMST
pip install -r requirements.txt



**`requirements.txt` contents:**
numpy
scikit-learn
matplotlib
optuna



### 3. Running the Notebooks

1.  Open your preferred Jupyter environment (e.g., Jupyter Lab, VS Code).
2.  Navigate to the cloned repository folder.
3.  Open any of the dataset notebooks (e.g., `HyEMST - JAIN.ipynb`).
4.  Run all cells in the notebook. The script will:
    -   Load the dataset.
    -   Run Bayesian hyperparameter optimization using **Optuna** to find the best parameters.
    -   Print the best parameters found.
    -   Display visualizations of the final clustering result.
    -   Print a comprehensive table of evaluation metrics (NMI, AMI, ARI, Silhouette, etc.).

## Citation

If you use HyEMST in your research, please cite our paper:

@article{Eyvazi2026HyEMST,
title={HyEMST: A Hybrid Density-Distance Clustering Algorithm with Maximum Spanning Trees for Non-Convex Geometries},
author={Eyvazi, Hossein and Rajaei, Ali},
....
}


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
it contains dataset dir too
note that noisy circle read from kaggle
but rest of them are txt files
final column is the label of them


Attachment
image.jpg
Here is an updated, precise README.md that includes the datasets directory and the data format details you mentioned.


# HyEMST: Hybrid Density-Distance Clustering with Maximum Spanning Trees

This repository contains the official implementation of **HyEMST**, a hybrid density–distance clustering algorithm based on Maximum Spanning Trees (MST) for discovering non-convex clusters with heterogeneous densities.

HyEMST decomposes data into micro-clusters, estimates volumetric densities via Minimum Volume Enclosing Ellipsoids (MVEE), builds a hybrid affinity kernel, extracts an MST, and performs adaptive density-aware merging to obtain the final clusters.

---

## Repository Structure

HyEMST/
├─ notebooks/
│ ├─ HyEMST - JAIN.ipynb
│ ├─ HyEMST - AGGREGATION.ipynb
│ ├─ HyEMST - R15.ipynb
│ ├─ HyEMST - D31.ipynb
│ └─ HyEMST - NOISY CIRCLES.ipynb
├─ datasets/
│ ├─ jain.txt
│ ├─ Aggregation.txt
│ ├─ R15.txt
│ └─ D31.txt
├─ requirements.txt
└─ README.md



- `notebooks/HyEMST - *.ipynb`: End‑to‑end experiments for each dataset (loading, Optuna optimization, MST construction, adaptive merging, visualizations, and metrics).
- `datasets/*.txt`: Benchmark datasets used in the paper.
- `requirements.txt`: Python dependencies.

---

## Datasets

### TXT Datasets (`datasets/`)

The following benchmark datasets are stored as plain text files:

- `jain.txt`
- `Aggregation.txt`
- `R15.txt`
- `D31.txt`

Each file has the same structure:

- Columns **1..d**: feature coordinates (for these benchmarks, `d = 2`).
- **Last column**: integer ground‑truth cluster label.

Example loading pattern (used in all notebooks):

import numpy as np

data = np.loadtxt("datasets/jain.txt")
X = data[:, :2] # features
y_true = data[:, 2].astype(int) # ground-truth labels



### Noisy Circles

The **Noisy Circles** dataset is not stored as a local `.txt` file.  
Instead, it is loaded/generated directly inside `HyEMST - NOISY CIRCLES.ipynb` using a Kaggle source / scikit-learn generator (see the first cells of that notebook). No manual download is required beyond running the notebook.

---

## Installation

pip install -r requirements.txt


`requirements.txt` should contain at least:

numpy
scikit-learn
matplotlib
optuna



---

## Running an Experiment

1. Open the desired notebook in `notebooks/` (e.g., `HyEMST - JAIN.ipynb`).
2. Make sure the corresponding dataset file exists under `datasets/`.
3. Run all cells:
   - The notebook will load the data.
   - Perform Bayesian hyperparameter optimization with Optuna.
   - Construct the MST and run adaptive merging.
   - Produce visualizations and print comprehensive clustering metrics (NMI, AMI, ARI, FMI, Silhouette, DB, CH, etc.).
