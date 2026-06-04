import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
from anndata import AnnData
from .graph import make_knn_graph_bluster
import warnings
from sklearn.cluster import KMeans
from .hw_kmeans import hw_kmeans
from .rng import coerce_rng
from .louvain_controlled import controlled_louvain_labels


def _r_style_kmeans(X, n_clusters, n_start=3, max_iter=50, random_state=0, trace=None):
    """Run k-means using R-like random initialization: sample rows as centers.

    We perform `n_start` restarts by sampling `n_clusters` distinct rows
    from `X` as initial centers (using CentralRNG) and keep the run with
    lowest inertia. This more closely matches R's `stats::kmeans` init.
    """
    rng = coerce_rng(random_state=random_state, rng=random_state)
    best_inertia = None
    best_labels = None
    best_centers = None

    X_arr = np.asarray(X)
    n_samples = X_arr.shape[0]

    for i in range(max(1, int(n_start))):
        # Sample distinct row indices for initial centers using R-style sample.int
        # `sample_int` returns 1-based indices, so convert to 0-based.
        init_idx = (rng.sample_int(n_samples, n_clusters, replace=False) - 1).astype(int)
        init_centers = X_arr[init_idx]

        # Use Hartigan-Wong style k-means implemented in hw_kmeans
        # Pass the sampled init indices so the initialization matches R's
        labels_hw, centers_hw = hw_kmeans(
            X_arr, n_clusters=n_clusters, n_start=1, max_iter=max_iter, rng=rng, init_idx=init_idx, trace=trace
        )
        # compute inertia
        inertia = np.sum((X_arr - centers_hw[labels_hw])**2)
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels_hw.copy()
            best_centers = centers_hw.copy()

    return best_labels, best_centers


def _top_variance_gene_indices(X, n_top):
    if sp.issparse(X):
        X = X.tocsr()
        means = np.asarray(X.mean(axis=0)).ravel()
        mean_squares = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        variances = mean_squares - means ** 2
    else:
        variances = np.var(X, axis=0)

    variances = np.asarray(variances).ravel()
    if n_top >= variances.size:
        return np.arange(variances.size)
    return np.argsort(variances)[::-1][:n_top]


def _louvain_labels(connectivities, random_state=0):
    return controlled_louvain_labels(connectivities, random_state=random_state)

def fast_cluster(adata, n_clusters=None, n_components=30, n_features=1000, 
                 key_added='clusters', use_gpu=False, random_state=0, rng=None, verbose=True, kmeans_trace=None):
    """
    Performs fast two-step clustering: K-means then graph clustering on centroids.

    This function mimics the logic of `fastcluster` in scDblFinder (R package).
    1. Runs K-means with a large k (e.g. 2500) on PCA components.
    2. Aggregates cells into centroids based on K-means clusters.
    3. Builds a KNN graph on these centroids.
    4. Runs graph clustering on the centroid graph.
    5. Propagates the labels back to original cells.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_clusters : int, optional
        Number of clusters for the initial K-means step. 
        If None, defaults to min(2500, n_cells/10).
    n_components : int, optional
        Number of PCA components to use.
    n_features : int, optional
        Number of highest-variance genes to use for PCA if not already present.
    key_added : str, optional
        Key under which to add the cluster labels in `adata.obs`.
    use_gpu : bool, optional
        Whether to use GPU acceleration via rapids-singlecell and cuML.
    random_state : int, optional
        Random seed.
    verbose : bool, optional
        Whether to print progress.
    kmeans_trace : list | callable | None, optional
        Optional trace sink forwarded to the CPU k-means step. When supplied,
        it receives dataset-agnostic event dicts describing start order, moves,
        and reseeding, which can be compared across datasets.
        
    Returns
    -------
    None
        Updates `adata.obs[key_added]` with the cluster labels.
    """
    
    if use_gpu:
        try:
            import rapids_singlecell as rsc  # type: ignore[import-not-found]
            import cuml  # type: ignore[import-not-found]
        except ImportError:
            warnings.warn("rapids-singlecell or cuml not found. Falling back to CPU.")
            use_gpu = False

    n_cells = adata.n_obs
    # Prefer a provided CentralRNG instance; coerce integers or None as needed
    rng = coerce_rng(random_state=random_state, rng=rng)
    
    # Handle missing X
    if adata.X is None and 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
        
    # R-style preprocessing: library-size normalize, then log-transform before PCA.
    # This mirrors scuttle::normalizeCounts followed by scater::runPCA more closely
    # than scanpy's default normalize_total/log1p path.
    if verbose: print("Checking normalization...")
    adata_calc = adata.copy()
    if sp.issparse(adata_calc.X):
        counts = adata_calc.X.toarray().astype(float)
    else:
        counts = np.asarray(adata_calc.X, dtype=float)

    lib_sizes = counts.sum(axis=1)
    mean_lib = lib_sizes.mean() if lib_sizes.size > 0 else 1.0
    size_factors = lib_sizes / mean_lib
    size_factors[size_factors == 0] = 1.0
    counts = np.log2(counts / size_factors[:, None] + 1.0)
    adata_calc.X = counts
    
    # Check for PCA
    if 'X_pca' not in adata_calc.obsm or adata_calc.obsm['X_pca'].shape[1] < n_components:
        if verbose: print("Selecting top variance genes for PCA...")
        top_gene_idx = _top_variance_gene_indices(adata_calc.X, min(n_features, adata_calc.n_vars))
        adata_calc = adata_calc[:, top_gene_idx].copy()

        if verbose: print("Running PCA...")
        if use_gpu:
            rsc.pp.pca(adata_calc, n_comps=n_components, random_state=int(rng.randint32()))
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_components, min(adata_calc.X.shape)), svd_solver="full")
            adata_calc.obsm["X_pca"] = pca.fit_transform(adata_calc.X)
             
    X_pca = adata_calc.obsm['X_pca'][:, :n_components]
    
    # Ensure X_pca is on CPU for K-means fallback or just generic compatibility if needed
    # However, cuml KMeans expects GPU data usually.
    # But later we use pandas.
    
    # 1. Determine k for K-means
    if n_clusters is None:
        k = min(2500, int(n_cells / 10))
    else:
        k = n_clusters
        
    # If cell count is small, just skip straight to graph clustering or use 1 cluster?
    # R logic: if nrow(x) > k (and > 1000 for creating metacells), do K-means.
    # Otherwise k is seq_len(nrow(x)) i.e. every cell is a cluster
    
    if n_cells > k and n_cells > 1000:
        if verbose: print(f"Running K-means with k={k}...")
        
        if use_gpu:
            kmeans_model = cuml.KMeans(n_clusters=k, random_state=random_state)
            # cuml usually expects GPU array (cupy) or numpy.
            # If X_pca is numpy, cuml handles it (copies to GPU).
            # If X_pca is cupy, cuml handles it.
            kmeans_labels = kmeans_model.fit_predict(X_pca)
            
            # For aggregation step (pandas), we need CPU arrays
            if hasattr(kmeans_labels, 'to_numpy'):
                 kmeans_labels_cpu = kmeans_labels.to_numpy()
            elif hasattr(kmeans_labels, 'get'): # cupy
                 kmeans_labels_cpu = kmeans_labels.get()
            else:
                 kmeans_labels_cpu = kmeans_labels
                 
            if hasattr(X_pca, 'get'):
                X_pca_cpu = X_pca.get()
            elif hasattr(X_pca, 'to_numpy'):
                X_pca_cpu = X_pca.to_numpy()
            else:
                X_pca_cpu = X_pca
        else:
            # Use R-style initialization: sample rows as initial centers and
            # pick the best of several starts. This makes the initialization
            # and randomness come from CentralRNG and better matches R.
            kmeans_labels, kmeans_centers = _r_style_kmeans(
                X_pca, n_clusters=k, n_start=10, max_iter=100, random_state=rng, trace=kmeans_trace
            )
            kmeans_labels_cpu = kmeans_labels
            X_pca_cpu = X_pca
            
        # 2. Aggregate to centroids
        # Using pandas for fast groupby mean
        df_pca = pd.DataFrame(X_pca_cpu)
        df_pca['label'] = kmeans_labels_cpu
        centroids = df_pca.groupby('label').mean().values
        unique_labels = np.sort(np.unique(kmeans_labels_cpu))
        
        # 3. Build KNN/SNN graph on centroids using bluster-like procedure
        if verbose: print("Building KNN/SNN graph on centroids...")
        n_centroids = centroids.shape[0]
        n_neighbors = min(max(2, int(np.sqrt(n_centroids)) - 1), 10)
        conn = make_knn_graph_bluster(centroids, k=n_neighbors, metric='euclidean', mode='knn')
        centroid_clusters = _louvain_labels(conn, random_state=rng)
            
        # 4. Map back to cells
        # Create a mapping dictionary: kmeans_label -> louvain_label
        # If groupby sorts by label (default is True), centroids are ordered by unique_labels
        
        map_dict = dict(zip(unique_labels, centroid_clusters))
        
        # Use kmeans_labels_cpu for mapping (safe for both CPU/GPU paths)
        final_clusters = np.array([map_dict[l] for l in kmeans_labels_cpu])
        
        adata.obs[key_added] = pd.Categorical(final_clusters.astype(str))
        
    else:
        # Fallback for small datasets: direct graph clustering
        if verbose: print("Running direct graph clustering (dataset small)...")
        n_neighbors = min(max(2, int(np.sqrt(n_cells)) - 1), 10)
        
        if use_gpu:
            rsc.pp.neighbors(adata_calc, n_neighbors=n_neighbors, use_rep='X_pca')
        else:
            sc.pp.neighbors(adata_calc, n_neighbors=n_neighbors, use_rep='X_pca')

        direct_clusters = _louvain_labels(adata_calc.obsp['connectivities'], random_state=rng)
        adata.obs[key_added] = pd.Categorical(direct_clusters.astype(str))

    if verbose:
        n_found = len(adata.obs[key_added].unique())
        print(f"Fast clustering found {n_found} clusters.")
          