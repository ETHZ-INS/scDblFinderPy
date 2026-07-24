from .rng import coerce_rng
from .misc import get_gpu_backend, to_backend_device, to_cpu


def fast_cluster(adata, n_components=30, n_features=1000, key_added='clusters',
                  use_gpu=False, random_state=0, rng=None, verbose=True, resolution=1.0):
    """
    Clusters cells using a standard scanpy / rapids-singlecell pipeline:
    normalize -> log1p -> HVG selection -> PCA -> neighbor graph -> Leiden.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_components : int
        Number of PCA components to use for the neighbor graph.
    n_features : int, optional
        Number of highly variable genes to select before PCA. If None or the
        dataset already has fewer genes, no HVG filtering is applied.
    key_added : str
        Key under which to add the cluster labels in `adata.obs`.
    use_gpu : bool
        Whether to use GPU acceleration via rapids-singlecell.
    random_state : int, optional
        Random seed (used if `rng` is not provided).
    rng : numpy.random.Generator, optional
        Shared RNG driving this pipeline's random draws.
    verbose : bool
        Whether to print progress.
    resolution : float
        Leiden clustering resolution.

    Returns
    -------
    None
        Updates `adata.obs[key_added]` with the cluster labels.
    """
    rng = coerce_rng(random_state=random_state, rng=rng)
    backend, seed = get_gpu_backend(use_gpu, rng)

    if adata.X is None and 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()

    adata_calc = adata.copy()
    if 'counts' not in adata_calc.layers:
        adata_calc.layers['counts'] = adata_calc.X.copy()
    to_backend_device(adata_calc, backend)

    if verbose: print("Normalizing counts...")
    backend.pp.normalize_total(adata_calc)
    backend.pp.log1p(adata_calc)

    if n_features is not None and adata_calc.n_vars > n_features:
        if verbose: print(f"Selecting top {n_features} variable genes...")
        backend.pp.highly_variable_genes(adata_calc, n_top_genes=n_features)
        adata_calc = adata_calc[:, adata_calc.var['highly_variable']].copy()

    if verbose: print("Running PCA...")
    n_comp_eff = min(n_components, min(adata_calc.shape) - 1)
    backend.pp.pca(adata_calc, n_comps=n_comp_eff, random_state=seed)

    if verbose: print("Building neighbor graph...")
    n_neighbors = min(15, adata_calc.n_obs - 1)
    backend.pp.neighbors(adata_calc, n_neighbors=n_neighbors, random_state=seed)

    if verbose: print("Running Leiden clustering...")
    backend.tl.leiden(adata_calc, resolution=resolution, random_state=seed, key_added=key_added)
    to_cpu(adata_calc, backend)

    adata.obs[key_added] = adata_calc.obs[key_added].astype(str).values

    if verbose:
        n_found = adata.obs[key_added].nunique()
        print(f"Fast clustering found {n_found} clusters.")
