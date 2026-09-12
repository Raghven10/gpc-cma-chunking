# Geometric Partition Cohesion & Contextual Manifold Alignment

Official code and experiment artifacts for the paper:  
**"Geometric Partition Cohesion and Contextual Manifold Alignment: Formal Metric Validation for Semantic Chunking in Long-Context Retrieval"**

---

## 📌 Overview

This repository provides the reference implementation and experimental pipeline for evaluating document chunk partitions in Retrieval-Augmented Generation (RAG) systems using intrinsic mathematical metrics:

1. **Intra-Chunk Semantic Density ($\text{ISD}$)**: Quantifies internal semantic compactness while regularizing pairwise semantic variance.
2. **Inter-Chunk Boundary Isolation ($\text{IBI}$)**: Measures normalized angular distance between adjacent chunk centroids to enforce sharp, non-redundant transitions.
3. **Geometric Partition Cohesion ($\text{GPC}$)**: Outlier-sensitive composite cohesion computed via a smoothed harmonic mean across chunks.
4. **Contextual Manifold Alignment ($\text{CMA}$)**: Measures the retention of document-level semantic subspace energy via centered Principal Component Analysis (PCA) projection.
5. **Contextual Distributional Fidelity ($\text{CDF}$)**: Quantifies unigram token distribution preservation via Jensen-Shannon Divergence ($\text{JSD}$).

---

## 📂 Repository Structure

```text
├── experiments/
│   ├── metrics.py                    # Formal definitions of ISD, IBI, GPC, CMA, CDF
│   ├── chunking_policies.py          # 8 partition strategies & null controls
│   ├── dataset_loader.py             # Qasper & multi-section Wikipedia loader
│   ├── run_construct_validity.py     # Experiment A: Multi-genre construct validity
│   ├── run_predictive_validity.py    # Experiment B: QA evidence retrieval benchmark
│   ├── generate_plots_and_tables.py  # Script to generate Tables 1-4 & Figures 1-3
│   ├── tests/
│   │   └── test_metrics.py           # Unit tests for boundary conditions & proofs
│   └── results/                      # Output CSV tables & high-res PNG figures
│       ├── construct_validity_raw.csv
│       ├── predictive_validity_raw.csv
│       ├── table1_protocol.csv
│       ├── table2_construct_validity_summary.csv
│       ├── table2_paired_hypothesis_tests.csv
│       ├── table3_qa_retrieval_summary.csv
│       ├── table4_predictive_validity_correlations.csv
│       ├── fig1_distributions.png
│       ├── fig2_pareto.png
│       └── fig3_sensitivity.png
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart & Setup

### 1. Installation
Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Raghven10/gpc-cma-chunking.git
cd gpc-cma-chunking

# Using uv (fastest) or standard python venv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run Unit Tests
Verify mathematical boundary conditions, single-sentence chunks, monotonicity, and rank limits:

```bash
python -m unittest discover -s experiments/tests
```

### 3. Run Construct Validity (Experiment A)
Evaluates 56 multi-genre documents across 8 chunking strategies and null controls:

```bash
python experiments/run_construct_validity.py
```

### 4. Run Predictive Validity (Experiment B)
Evaluates evidence retrieval on Qasper QA across matched retrieved-token budgets ($B \in \{256, 512, 1024\}$ tokens):

```bash
python experiments/run_predictive_validity.py
```

### 5. Generate Figures and Tables
Regenerate publication-quality charts and summary CSV tables:

```bash
python experiments/generate_plots_and_tables.py
```

---

## 📊 Summary of Experimental Results

### Construct Validity (Reference Structure vs. Null Controls)
- **Gold vs. ShuffledNull on $\text{GPC}$**: $\Delta = +0.0441$, $p = 2.86 \times 10^{-35}$, **Cohen's $d = 3.920$**
- **Gold vs. ShuffledNull on $\text{CMA}$**: $\Delta = +0.1387$, $p = 5.61 \times 10^{-31}$, **Cohen's $d = 3.231$**
- **Gold vs. RandomMatched on $\text{CMA}$**: $\Delta = +0.0197$, $p = 2.40 \times 10^{-5}$, **Cohen's $d = 0.617$**

### QA Evidence Retrieval Benchmark ($B = 512$ Tokens)
- `Recursive-256` achieves the highest evidence coverage ($51.7\%$) and recall ($0.628$) under fixed token budgets.
- `Semantic-75` maximizes internal geometric cohesion ($\text{GPC} = 0.582$).

