# scDblFinderPy

Python implementation of the
[scDblFinder](https://github.com/plger/scDblFinder) workflow for doublet
detection in single-cell RNA-seq data, designed to run on AnnData/Scanpy
objects. This package reimplements the core algorithms of that R/Bioconductor
package in Python; see the original repository for the reference
implementation and further methodological background.

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

## Setup

There are two independent installation paths below, depending on whether you
need GPU acceleration:
- **CPU-only:** a plain `venv`, described in
  [Install (CPU, via `venv`)](#install-cpu-via-venv).
- **GPU (`use_gpu=True`):** a `conda` environment, described in
  [Optional — GPU acceleration (via `conda`)](#optional--gpu-acceleration-via-conda).
  This is
  required because the CUDA packages (`cuml`, `cugraph`, etc.) are only
  distributed via conda, not pip/venv.

Pick one based on your needs — don't mix the two (e.g. don't try to `pip
install` the GPU stack into the venv from the CPU path).

### Requirements

Python **3.12 or later**. The pinned dependency versions below (in
particular `scanpy==1.12.1` and `numpy==2.4.3`) don't publish wheels for
Python 3.10 or 3.11, so installing will fail on those versions with a
"Could not find a version that satisfies the requirement" error. Check your
interpreter before creating the virtual environment below:

```bash
python3 --version   # must be 3.12+
```

### Install (CPU, via `venv`)

Install directly from GitHub:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/ETHZ-INS/scDblFinderPy.git
```

This installs the exact dependency versions pinned in `pyproject.toml`
(`numpy`, `pandas`, `scipy`, `anndata`, `scanpy`, `scikit-learn`, `xgboost`,
`statsmodels`, `leidenalg`, `igraph`) and registers `scDblFinderPy` as a
normal importable package, plus the `scDblFinderPy` CLI command — no
`sys.path` hacks needed. The versions are pinned exactly, rather than left as
open ranges, because minor version differences in these libraries can shift
doublet scores enough to change calls near the threshold — pinning keeps
results reproducible across machines and over time.

If you want to modify the code, or you need a local checkout for the GPU
setup below, clone the repository and install from it instead:

```bash
git clone https://github.com/ETHZ-INS/scDblFinderPy.git
cd scDblFinderPy
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .   # editable: edits take effect without reinstalling
```

### Optional — GPU acceleration (via `conda`)

When `use_gpu=True`, clustering/PCA/neighbor search run via
`rapids-singlecell` (cuML/cuGraph under the hood) and XGBoost trains on the
GPU (`device='cuda'`).

**Do not try to `pip install rapids-singlecell cuml` into a plain venv** —
`cuml`, `cudf`, and `cugraph` are compiled CUDA extensions that NVIDIA
distributes through conda/mamba, matched to your CUDA toolkit and driver
version. A bare pip install of these will not work on most machines. The
supported path is: build the GPU stack with conda first, then layer this
package's pure-Python dependencies on top with pip — **not** into the venv
from the Install section above, since that would fight with conda over
which numpy/scipy/pandas build is active.

**Step 1 — create the conda environment with the RAPIDS GPU stack.** Use the
[RAPIDS release selector](https://docs.rapids.ai/install/) to get the exact
`conda create` command for your CUDA toolkit/driver version and Python
3.12+, selecting at least `cuml`, `cugraph`, and `cuvs` (this also pulls in
`cudf` and `cupy` as dependencies). `cuvs` provides the GPU nearest-neighbor
search backend `cuml` relies on, so it must be installed alongside `cuml`
and `cugraph` here. This package was developed and tested against:

```bash
conda create -n scdblfinder-gpu -c rapidsai -c conda-forge -c nvidia \
    cuml=26.4 cugraph=26.4 cuvs=26.4 python=3.12 'cuda-version>=12.0,<=13.2'
conda activate scdblfinder-gpu
```

During this step conda may print an "nvidia" channel notice about several
CUDA packages (`cuda`, `cudnn`, `nccl`, etc.) no longer receiving updates on
the `nvidia` channel after Aug 7, 2026, and pointing to the `defaults` or
`conda-forge` channels for future updates. This is an informational notice,
not an error — existing packages and environments keep working, and it does
not affect this install. It's only worth acting on if you hit missing
updates for those packages in a future environment.

**Step 2 — layer `rapids-singlecell` and this package's remaining
dependencies on top, via pip, inside that same conda environment.** Use
`--no-deps` for this package so pip doesn't try to reinstall numpy/scipy/
pandas over the versions conda just resolved for the CUDA stack:

```bash
# rapids-singlecell ships CUDA-version-suffixed PyPI builds; match it to the
# cuda-version conda just installed (e.g. -cu12 or -cu13):
pip install rapids-singlecell-cu13 xgboost==3.2.0
pip install --no-deps .
# or, for local development: pip install --no-deps -e .
pip install anndata==0.12.13 scanpy==1.12.1 scikit-learn==1.8.0 \
    statsmodels==0.14.6 leidenalg==0.11.0 igraph==1.0.0
```

These versions match the pins in `pyproject.toml` (see the note on pinning
in the [Install (CPU, via `venv`)](#install-cpu-via-venv) section above) —
installing
unpinned versions here risks pulling in newer releases that conflict with
what `cuml`/`rapids-singlecell` expect, or that shift doublet scores
relative to the tested configuration.

(`xgboost`'s regular PyPI wheel already includes GPU support via
`device='cuda'`; RAPIDS' own conda channel also ships a GPU-enabled build if
you prefer installing it alongside `cuml`/`cugraph` in step 1 instead.)

Verify the install picked up the GPU stack before running anything real:

```bash
python -c "import rapids_singlecell, cuml, cupy; print('GPU stack OK')"
```

This was developed and tested against `rapids-singlecell==0.15.0`,
`cuml==26.4.0`, `cudf==26.4.0`, `cugraph==26.4.0`, `cuvs==26.4.0`,
`cupy==14.0.1`.

## Command-line interface

Installing the package also registers a `scDblFinderPy` command, useful when
scoring many samples (e.g. in a shell loop) without writing a Python script
for each one:

```bash
# Write scores + class into the input file itself (adds/overwrites
# scDblFinder_score and scDblFinder_class in adata.obs):
scDblFinderPy --adata sample.h5ad

# Write scores to a separate CSV instead of touching the h5ad file:
scDblFinderPy --adata sample.h5ad --output doublet_scores.csv

# Batch over many samples:
for f in samples/*.h5ad; do
    scDblFinderPy --adata "$f" --output "scores/$(basename "$f" .h5ad).csv"
done

# Opt in to GPU mode (requires the RAPIDS/cuML setup from the GPU
# acceleration section above; runs on CPU by default otherwise):
scDblFinderPy --adata sample.h5ad --use-gpu
```

CPU is the default — the CLI only runs on GPU if you pass `--use-gpu`,
and doing so requires the conda-based RAPIDS stack described in
[Optional — GPU acceleration (via `conda`)](#optional--gpu-acceleration-via-conda)
to already be installed in the active environment.

Run `scDblFinderPy --help` for the full list of options (clustered mode,
number of iterations, random seed, etc.) — these mirror the
`compute_doublet_score(...)` parameters described below.

## Using the package in your own scripts

Once installed (see [Setup](#setup) above), import it directly — no
`sys.path` hacks required:

```python
from scDblFinderPy.scDblFinder import compute_doublet_score
```

### Input expectations

`compute_doublet_score(...)` expects an `AnnData` object where:
- `adata.X` contains raw counts (preferred), **or**
- `adata.layers['counts']` contains raw counts.

### Random mode

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

### Clustered mode

```python
adata_out = compute_doublet_score(
    adata,
    clusters_col="clusters",  # column is computed and stored here if absent
    n_iters=3,
    random_state=42,
    verbose=True,
)
```

## Outputs

In `adata.obs`:
- `scDblFinder_score` — continuous doublet score (higher = more likely doublet)
- `scDblFinder_class` — final call: `doublet` or `singlet`

In `adata.uns`:
- `scDblFinder_threshold` — the score threshold used for the final classification

If `return_type='full'`, the returned object also includes artificial doublets.

## Smoothing doublet scores

`smooth_doublet_scores(...)` is an optional post-hoc step, ported from R's
`smoothDoubletScores()`. Like its R counterpart, it is **not** called
automatically by `compute_doublet_score` — call it explicitly afterwards if
you want it. It blends each cell's doublet score (in logit space) with a
distance-weighted consensus of its neighbors in some embedding (PCA by
default), which amplifies coherent clusters of high-probability doublets
while suppressing isolated, noisy-looking predictions:

```python
from scDblFinderPy import compute_doublet_score, smooth_doublet_scores

adata_out = compute_doublet_score(adata, clusters_col=None, random_state=42)

# compute_doublet_score does not retain a PCA embedding on its output, so
# compute/attach one first (e.g. via scanpy) if you don't already have one:
import scanpy as sc
sc.pp.normalize_total(adata_out)
sc.pp.log1p(adata_out)
sc.pp.pca(adata_out, n_comps=20)

adata_out = smooth_doublet_scores(adata_out, coords="X_pca", k=20)
print(adata_out.obs[["scDblFinder_score", "smoothedDoubletScore"]].head())
```

It also works directly on a plain array of scores plus a coordinate matrix
(or a precomputed `{"index": ..., "distance": ...}` neighbor dict), without
needing an `AnnData`:

```python
from scDblFinderPy import smooth_doublet_scores

smoothed = smooth_doublet_scores(doublet_scores, coords=pca_coords, k=20)
```

Key parameters: `alpha` (0-1, blend between a cell's own score and its
neighborhood's) and `gamma` (>= 1, amplifies coherent high-probability
neighborhoods when > 1). See the function's docstring for the full
parameter list (`decay_by_distance`, `weights`, etc.).

## Notes and current limitations

- Multi-sample mode (R's `samples`/`multiSampleMode` arguments) is not
  implemented; there is no `samples_col`-equivalent parameter. All cells are
  processed together regardless of sample of origin.
- Some low-level numerical differences from the R package are expected due to
  library backend differences.
