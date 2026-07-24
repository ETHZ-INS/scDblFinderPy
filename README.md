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
3. Combined real + artificial embedding (PCA) and KNN feature extraction.
4. Iterative XGBoost training and score refinement.
5. Final thresholding to obtain doublet calls.

Clustering, PCA, and nearest-neighbor search are delegated to `scanpy`
(CPU) / `rapids-singlecell` (GPU, when `use_gpu=True`) rather than
reimplemented — only the doublet-detection-specific logic (artificial
doublet generation, CXDS scoring, KNN-derived doublet features, iterative
XGBoost training, threshold optimization) is custom to this package.

## Repository layout

```
scDblFinderPy/              ← repo root (this directory is the Python package)
├── pyproject.toml          package metadata + pinned dependency versions
├── scDblFinder.py          main pipeline — contains compute_doublet_score()
├── clustering.py           fast_cluster(): scanpy/rapids-singlecell clustering
├── doublet_generation.py
├── misc.py                 cxds2, select_features, GPU-backend helpers
├── thresholding.py
├── rng.py                  coerce_rng(): numpy Generator seeding helper
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

### 3. Install the package

```bash
pip install .
# or, for local development (edits take effect without reinstalling):
pip install -e .
```

This installs the exact dependency versions pinned in `pyproject.toml`
(`numpy`, `pandas`, `scipy`, `anndata`, `scanpy`, `scikit-learn`, `xgboost`,
`statsmodels`, `leidenalg`, `igraph`) and registers `scDblFinderPy` as a
normal importable package — no `sys.path` hacks needed. See
[Reproducibility](#reproducibility) below for why the pins matter.

Optional — GPU acceleration (requires a CUDA-capable machine). When
`use_gpu=True`, clustering/PCA/neighbor search run via `rapids-singlecell`
and XGBoost trains on the GPU (`device='cuda'`):

```bash
pip install .[gpu]
```

`rapids-singlecell` and `cuml` are not reliably pip-installable in general
(the available wheels depend on your exact CUDA toolkit version) — NVIDIA's
recommended path is conda/mamba via the
[RAPIDS release selector](https://docs.rapids.ai/install/). This package was
developed and tested against `rapids-singlecell==0.15.0`, `cuml==26.4.0`,
`cupy==14.0.1` installed that way.

## Using the package in your own scripts

Once installed (see Setup), import it directly:

```python
from scDblFinderPy.scDblFinder import compute_doublet_score
```

If you'd rather not install it — e.g. quick experimentation without
`pip install` — you can still fall back to adding the repo's **parent**
directory to `sys.path` (this skips the pinned-version guarantee below,
since it relies on whatever is already on `sys.path`):

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
| `n_features` | `1352` | number of genes used for feature selection (matches R's `nfeatures`) |
| `n_components` | `20` | number of PCA components |
| `n_artificial` | `None` | override number of artificial doublets (auto if `None`) |
| `prop_random` | `0` | fraction of artificial doublets generated randomly (matches R's `propRandom`) |
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

## Reproducibility

Given the same input data, `random_state`, and library versions, two runs on
the same machine produce bit-identical `scDblFinder_score` output (verified
for both `use_gpu=False` and `use_gpu=True`, in random and clustered mode).
That guarantee has some real limits worth knowing before you rely on it:

- **Pin your dependency versions.** `scanpy`, `leidenalg`, `xgboost`, and
  `numpy` all change numerical internals (PCA solver defaults, clustering
  backend, histogram construction) across releases, independent of RNG
  seeding. Installing via `pip install .` (see Setup) pins the exact
  versions this package was developed against — an unpinned
  `pip install scanpy ...` will drift over time.
- **Different machines are not guaranteed to match**, even with identical
  package versions. numpy/scipy delegate to BLAS (OpenBLAS/MKL), which picks
  different code paths per CPU (AVX2/AVX512/thread count), producing
  ULP-level floating-point differences that can cascade through PCA into
  clustering and scores. This is a general property of the numpy/scipy/scanpy
  stack, not specific to this package.
- **GPU mode (`use_gpu=True`) is best-effort, not guaranteed**, across
  different GPU models, driver versions, or CUDA toolkit versions — RAPIDS/
  cuML has known non-deterministic reduction operations in places. Treat
  GPU reproducibility as "consistent within one fixed environment," not
  "identical everywhere."
- Use the same preprocessing assumptions (counts in `adata.X` or
  `adata.layers['counts']`) when comparing runs.

## Notes and current limitations

- Multi-sample mode (R's `samples`/`multiSampleMode` arguments) is not
  implemented; there is no `samples_col`-equivalent parameter. All cells are
  processed together regardless of sample of origin.
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
