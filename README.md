# scDblFinderPy

Python implementation of the scDblFinder workflow for doublet detection in
single-cell RNA-seq data, designed to run on AnnData/Scanpy objects.

This module mirrors the core ideas of the R package:
- optional pre-clustering
- artificial doublet generation
- iterative classifier training
- threshold optimization based on expected doublet rate

## What this package does

Given a count matrix in an `AnnData` object, scDblFinderPy estimates a
doublet score for each real cell and returns a final class (`doublet` or
`singlet`).

At a high level, the pipeline is:
1. Optional clustering of real cells (clustered mode).
2. Feature selection and artificial doublet generation.
3. Combined real + artificial embedding and KNN feature extraction.
4. Iterative XGBoost training and score refinement.
5. Final thresholding to obtain doublet calls.

## Repository layout

```
scDblFinderPy/              ← repo root (this directory is the Python package)
├── scDblFinder.py          main pipeline — contains compute_doublet_score()
├── clustering.py
├── doublet_generation.py
├── misc.py
├── thresholding.py
├── rng.py
├── graph.py
├── biocneighbors_kmknn.py
├── louvain_controlled.py
├── hw_kmeans.py
├── r_mt19937.py
├── r_sample_emulation.py
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd scDblFinderPy
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install numpy pandas scipy anndata scanpy scikit-learn xgboost
```

Optional — GPU acceleration (requires a CUDA-capable machine):

```bash
pip install rapids-singlecell cuml
```

## Using the package in your own scripts

Because the repo root itself is the Python package, you need to add its
**parent directory** to `sys.path` before importing. Assuming you cloned into
`/path/to/scDblFinderPy`:

```python
import sys
sys.path.insert(0, "/path/to")   # parent of the scDblFinderPy/ directory
from scDblFinderPy.scDblFinder import compute_doublet_score
```

### Input expectations

`compute_doublet_score(...)` expects an `AnnData` object where:
- `adata.X` contains raw counts (preferred), **or**
- `adata.layers['counts']` contains raw counts.

### Random mode (no clustering)

```python
import scanpy as sc
import sys
sys.path.insert(0, "/path/to")
from scDblFinderPy.scDblFinder import compute_doublet_score

adata = sc.read_h5ad("your_data.h5ad")
adata_out = compute_doublet_score(
    adata,
    clusters_col=None,   # random mode — no clustering step
    n_iters=3,
    random_state=42,
    verbose=True,
)

print(adata_out.obs[["scDblFinder_score", "scDblFinder_class"]].head())
print("Threshold:", adata_out.uns.get("scDblFinder_threshold"))
```

### Clustered mode (auto clustering)

```python
adata_out = compute_doublet_score(
    adata,
    clusters_col="clusters",  # column is computed and stored here if absent
    n_iters=3,
    random_state=42,
    verbose=True,
)
```

### Clustered mode (precomputed clusters)

```python
adata.obs["my_clusters"] = ...   # your own cluster labels
adata_out = compute_doublet_score(adata, clusters_col="my_clusters")
```

## Outputs

In `adata.obs`:
- `scDblFinder_score` — continuous doublet score (higher = more likely doublet)
- `scDblFinder_class` — final call: `doublet` or `singlet`

In `adata.uns`:
- `scDblFinder_threshold` — the score threshold used for the final classification

If `return_type='full'`, the returned object also includes artificial doublets.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `clusters_col` | `None` | `None` for random mode; column name for clustered mode |
| `n_features` | `1352` | number of genes used for feature selection |
| `n_components` | `20` | number of PCA components |
| `n_artificial` | `None` | override number of artificial doublets (auto if `None`) |
| `prop_random` | `0.1` | fraction of artificial doublets generated randomly |
| `n_iters` | `3` | iterative classifier refinement rounds |
| `dbr_per1k` | `0.008` | expected doublet rate per 1k cells |
| `stringency` | `0.5` | threshold optimisation aggressiveness |
| `random_state` | `42` | reproducibility seed |
| `use_gpu` | `False` | enable GPU-accelerated steps (requires rapids/cuml) |
| `verbose` | `True` | print progress at each stage |

## Running the benchmarks

The benchmark scripts live in `benchmarking/` and must be run from **inside
that directory** so that relative dataset paths resolve correctly.

**Run all datasets:**

```bash
cd benchmarking
python run_python_benchmark.py
```

Results are saved to `benchmarking/python_benchmark_metrics.csv`.

**Run a single dataset** (e.g. `hm-6k`):

```bash
cd benchmarking
python run_dataset.py hm-6k
# optionally pass a repeat count: python run_dataset.py hm-6k 3
```

Results are saved to `benchmarking/python_benchmark_hm-6k.csv`.

Datasets must be present as `benchmarking/datasets/<name>.h5ad` and must
contain a `truth` column in `adata.obs` with values `doublet` / `singlet`.

## Reproducibility tips

- Fix `random_state` when comparing runs.
- Keep package versions stable (especially `scanpy`, `scikit-learn`, `xgboost`).
- Use the same preprocessing assumptions (counts in `adata.X` or `adata.layers['counts']`).

## Notes and current limitations

- `samples_col` is accepted but currently ignored in the Python pipeline.
- Some low-level numerical differences from the R package are expected due to
  library backend differences.

## Troubleshooting

**Results look unstable or weak:**
- Confirm counts are raw (not log-normalised or otherwise transformed).
- Try both modes (`clusters_col=None` and a clustered mode).
- Check `xgboost` version is compatible with your Python version.
- Run with `verbose=True` to inspect each stage.

**Clustering looks poor:**
- Pass your own precomputed cluster labels via `clusters_col` instead of relying
  on the built-in fast clustering.
