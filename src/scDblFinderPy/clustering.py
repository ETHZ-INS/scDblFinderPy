import numpy as np
import scipy.sparse as sp

from .rng import coerce_rng
from .misc import get_gpu_backend, to_backend_device, to_cpu


def _kmeans_labels(X, k, n_init=3, max_iter=50, seed=0, use_gpu=False):
    """
    k-means over `X` (n_units x n_dims), mirroring R's `kmeans(x, k, nstart=3, iter.max=50)`.

    R's default kmeans() algorithm is Hartigan-Wong; scikit-learn only offers
    Lloyd's algorithm (or the 'elkan' variant, same convergence properties).
    Raising n_init well past 3 makes the resulting meta-cell partition more
    balanced and more similar to R's own k-means output in isolation (checked
    directly against R's actual preclusters/ARI on real data), but empirically
    made end-to-end AUPRC *worse* on both pbmc-1A and pbmc-1B compared to
    n_init=3 — so matching R's nstart=3 literally, despite the local k-means
    mismatch, is kept as the better-performing choice in practice.
    """
    if use_gpu:
        try:
            import cuml
            # cuML's default init='scalable-k-means++' is an approximate
            # initialization designed for distributed/huge-scale data; on the
            # small precluster counts used here it's a genuinely different
            # (noisier) algorithm than sklearn's exact 'k-means++' below, not
            # just a numerically-noisier version of the same one. Force the
            # same exact algorithm sklearn uses so the two backends solve the
            # same clustering problem.
            km = cuml.cluster.KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                                      random_state=seed, init='k-means++')
            return np.asarray(km.fit_predict(X)).astype(int)
        except Exception:
            pass
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed)
    return km.fit_predict(X).astype(int)


def _louvain_labels(X, n_units, resolution=1.0, graph_k=None):
    """
    Build a small KNN graph over `X` (n_units x n_dims) and cluster it with
    Louvain community detection, mirroring R's
    `bluster::makeKNNGraph(x, k=min(max(2,floor(sqrt(n)))-1),10))` followed by
    `igraph::cluster_louvain`. `igraph`'s `community_multilevel` is the same
    Blondel et al. algorithm R's `igraph::cluster_louvain` calls, so this
    reproduces that step without needing a separate Louvain package.
    """
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    if n_units <= 1:
        return np.zeros(n_units, dtype=int)

    if graph_k is None:
        graph_k = min(max(2, int(np.floor(np.sqrt(n_units))) - 1), 10)
    graph_k = max(1, min(graph_k, n_units - 1))

    nn = NearestNeighbors(n_neighbors=graph_k + 1).fit(X)
    _, neighbor_idx = nn.kneighbors(X)
    edges = set()
    for i in range(n_units):
        for j in neighbor_idx[i, 1:]:  # skip self (column 0)
            j = int(j)
            edges.add((i, j) if i < j else (j, i))

    g = ig.Graph(n=n_units, edges=list(edges))
    membership = g.community_multilevel(resolution=resolution).membership
    return np.asarray(membership, dtype=int)


def fast_cluster(adata, n_components=30, n_features=1000, key_added='clusters',
                  use_gpu=False, random_state=0, rng=None, verbose=True, resolution=1.0):
    """
    Clusters cells using the same two-step strategy as R's `fastcluster`:
    normalize -> log1p -> top-variance gene selection -> PCA, then a k-means +
    Louvain pre-clustering step (rather than Leiden on the full per-cell
    graph).

    For >1000 cells, this mirrors `fastcluster`'s meta-cell approach: cells are
    first split into `min(2500, n//10)` fine-grained k-means groups, each
    group's PCA coordinates are averaged into a meta-cell, a small KNN graph
    is built over the meta-cells, and Louvain community detection
    (`igraph::cluster_louvain` in R; `igraph`'s `community_multilevel` here)
    assigns each meta-cell to a cluster. Cluster labels are then propagated
    back from meta-cells to the individual cells they were built from. For
    <=1000 cells, Louvain runs directly on the per-cell KNN graph (no
    meta-cell step), matching R's small-dataset fallback.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_components : int
        Number of PCA components to use for the neighbor graph.
    n_features : int, optional
        Number of top-variance genes to select before PCA. If None or the
        dataset already has fewer genes, no feature filtering is applied.
    key_added : str
        Key under which to add the cluster labels in `adata.obs`.
    use_gpu : bool
        Whether to use GPU acceleration for normalization/PCA (via
        rapids-singlecell) and for the k-means pre-clustering step (via
        cuML, if installed; falls back to scikit-learn otherwise). The
        meta-cell KNN graph is small regardless of dataset size, so the
        Louvain step always runs on CPU via `igraph`.
    random_state : int, optional
        Random seed (used if `rng` is not provided).
    rng : numpy.random.Generator, optional
        Shared RNG driving this pipeline's random draws.
    verbose : bool
        Whether to print progress.
    resolution : float
        Louvain resolution parameter (passed to `igraph`'s
        `community_multilevel`).

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
        if verbose: print(f"Selecting top {n_features} variable genes (by variance)...")
        # R's fastcluster feeds scater::runPCA(ntop=nfeatures), which ranks genes by
        # raw variance of the log-normalized expression and takes the top `ntop` —
        # not scanpy's mean-binned dispersion HVG method. Match that plain ranking
        # here, since it's this PCA (not the main scoring PCA) that determines the
        # cluster assignment.
        X = adata_calc.X
        if hasattr(X, 'get'):  # cupy / cupyx sparse -> back to CPU for this reduction
            X = X.get()
        if sp.issparse(X):
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            n = X.shape[0]
            var = (mean_sq - mean ** 2) * (n / (n - 1)) if n > 1 else mean_sq - mean ** 2
        else:
            var = np.var(np.asarray(X), axis=0, ddof=1)
        # Stable descending sort, matching R's order(x, decreasing=TRUE) tie-breaking.
        top_idx = np.argsort(-var, kind='stable')[:n_features]
        adata_calc = adata_calc[:, top_idx].copy()

    if verbose: print("Running PCA...")
    n_comp_eff = min(n_components, min(adata_calc.shape) - 1)
    backend.pp.pca(adata_calc, n_comps=n_comp_eff, random_state=seed)
    to_cpu(adata_calc, backend)

    X_pca = np.asarray(adata_calc.obsm['X_pca'])
    n_cells = X_pca.shape[0]

    # R: k <- min(2500, floor(nrow(x)/10))
    k_pre = min(2500, n_cells // 10)

    # R: if((returnType != "clusters" || nrow(x)>1000) && nrow(x)>k) build meta-cells;
    # for our fixed returnType="clusters" this reduces to nrow(x)>1000 && nrow(x)>k.
    if n_cells > 1000 and n_cells > k_pre and k_pre >= 2:
        if verbose: print(f"Building {k_pre} meta-cells (k-means pre-clustering)...")
        precluster = _kmeans_labels(X_pca, k_pre, seed=seed, use_gpu=use_gpu)
        group_ids = np.unique(precluster)
        meta_X = np.vstack([X_pca[precluster == g].mean(axis=0) for g in group_ids])
        group_pos = {g: i for i, g in enumerate(group_ids)}
        precluster_pos = np.array([group_pos[g] for g in precluster])
        n_groups = len(group_ids)
    else:
        precluster_pos = np.arange(n_cells)
        meta_X = X_pca
        n_groups = n_cells

    if verbose: print("Building KNN graph over meta-cells and running Louvain clustering...")
    meta_membership = _louvain_labels(meta_X, n_groups, resolution=resolution)

    labels = meta_membership[precluster_pos]
    adata.obs[key_added] = labels.astype(str)

    if verbose:
        n_found = adata.obs[key_added].nunique()
        print(f"Fast clustering found {n_found} clusters.")
