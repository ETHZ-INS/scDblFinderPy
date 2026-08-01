# scDblFinderPy

Python implementation of the
[scDblFinder](https://github.com/plger/scDblFinder) workflow for doublet
detection in single-cell RNA-seq data, designed to run on AnnData/Scanpy
objects. This package reimplements the core algorithms of that R/Bioconductor
package in Python; see the original repository for the reference
implementation and further methodological background.

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
scDblFinderPy/                      ← repo root
├── pyproject.toml                  package metadata + pinned dependency versions
├── src/scDblFinderPy/               importable package (src layout)
│   ├── scDblFinder.py               main pipeline — contains compute_doublet_score()
│   ├── clustering.py                fast_cluster(): scanpy/rapids-singlecell clustering
│   ├── doublet_generation.py
│   ├── misc.py                      cxds2, select_features, GPU-backend helpers
│   ├── thresholding.py
│   ├── rng.py                       coerce_rng(): numpy Generator seeding helper
│   └── __init__.py
```

## Setup

### Requirements

Python **3.12 or later**. The pinned dependency versions below (in
particular `scanpy==1.12.1` and `numpy==2.4.3`) don't publish wheels for
Python 3.10 or 3.11, so `pip install .` will fail on those versions with a
"Could not find a version that satisfies the requirement" error. Check your
interpreter before creating the virtual environment in step 2:

```bash
python3 --version   # must be 3.12+
```

### 1. Clone the repository

```bash
git clone <repo-url>
cd scDblFinderPy
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
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

### 4. Optional — GPU acceleration

When `use_gpu=True`, clustering/PCA/neighbor search run via
`rapids-singlecell` (cuML/cuGraph under the hood) and XGBoost trains on the
GPU (`device='cuda'`).

**Do not try to `pip install rapids-singlecell cuml` into a plain venv** —
`cuml`, `cudf`, and `cugraph` are compiled CUDA extensions that NVIDIA
distributes through conda/mamba, matched to your CUDA toolkit and driver
version. A bare pip install of these will not work on most machines. The
supported path is: build the GPU stack with conda first, then layer this
package's pure-Python dependencies on top with pip — **not** into the venv
from steps 2–3 above, since that would fight with conda over which
numpy/scipy/pandas build is active.

**Step 1 — create the conda environment with the RAPIDS GPU stack.** Use the
[RAPIDS release selector](https://docs.rapids.ai/install/) to get the exact
`conda create` command for your CUDA toolkit/driver version and Python
3.12+, selecting at least `cuml` and `cugraph` (this also pulls in `cudf`
and `cupy` as dependencies). This package was developed and tested against:

```bash
conda create -n scdblfinder-gpu -c rapidsai -c conda-forge -c nvidia \
    cuml=26.4 cugraph=26.4 python=3.12 'cuda-version>=12.0,<=13.2'
conda activate scdblfinder-gpu
```

**Step 2 — layer `rapids-singlecell` and this package's remaining
dependencies on top, via pip, inside that same conda environment.** Use
`--no-deps` for this package so pip doesn't try to reinstall numpy/scipy/
pandas over the versions conda just resolved for the CUDA stack:

```bash
# rapids-singlecell ships CUDA-version-suffixed PyPI builds; match it to the
# cuda-version conda just installed (e.g. -cu12 or -cu13):
pip install rapids-singlecell-cu13 xgboost
pip install --no-deps .
# or, for local development: pip install --no-deps -e .
pip install anndata scanpy scikit-learn statsmodels leidenalg igraph
```

(`xgboost`'s regular PyPI wheel already includes GPU support via
`device='cuda'`; RAPIDS' own conda channel also ships a GPU-enabled build if
you prefer installing it alongside `cuml`/`cugraph` in step 1 instead.)

Verify the install picked up the GPU stack before running anything real:

```bash
python -c "import rapids_singlecell, cuml, cupy; print('GPU stack OK')"
```

This was developed and tested against `rapids-singlecell==0.15.0`,
`cuml==26.4.0`, `cudf==26.4.0`, `cugraph==26.4.0`, `cupy==14.0.1`.

## Using the package in your own scripts

Install directly from GitHub — no local clone needed:

```bash
pip install git+https://github.com/ETHZ-INS/scDblFinderPy.git
```

Or, if you've already cloned it locally (see Setup), `pip install .` /
`pip install -e .` from the repo root registers the same importable
package. Either way, import it directly — no `sys.path` hacks required:

```python
from scDblFinderPy.scDblFinder import compute_doublet_score
```

### Input expectations

`compute_doublet_score(...)` expects an `AnnData` object where:
- `adata.X` contains raw counts (preferred), **or**
- `adata.layers['counts']` contains raw counts.

### Random mode (no clustering)

```python
import scanpy as sc
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
