import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
from anndata import AnnData
import warnings
from sklearn.cluster import KMeans

def fast_cluster(adata, n_clusters=None, n_components=30, n_features=1000, 
                 key_added='clusters', use_gpu=False, random_state=0, verbose=True):
    """
    Performs fast two-step clustering: K-means then Leiden on centroids.
    
    This function mimics the logic of `fastcluster` in scDblFinder (R package).
    1. Runs K-means with a large k (e.g. 2500) on PCA components.
    2. Aggregates cells into centroids based on K-means clusters.
    3. Builds a KNN graph on these centroids.
    4. Runs Leiden clustering on the graph (replaces Louvain for better stability).
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
        Number of highly variable genes to use for PCA if not already present.
    key_added : str, optional
        Key under which to add the cluster labels in `adata.obs`.
    use_gpu : bool, optional
        Whether to use GPU acceleration via rapids-singlecell and cuML.
    random_state : int, optional
        Random seed.
    verbose : bool, optional
        Whether to print progress.
        
    Returns
    -------
    None
        Updates `adata.obs[key_added]` with the cluster labels.
    """
    
    if use_gpu:
        try:
            import rapids_singlecell as rsc
            import cuml
        except ImportError:
            warnings.warn("rapids-singlecell or cuml not found. Falling back to CPU.")
            use_gpu = False

    n_cells = adata.n_obs
    
    # Handle missing X
    if adata.X is None and 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
        
    # Preprocessing to avoid issues with raw counts in PCA
    # scDblFinder R does logNormCounts before PCA
    # We should do the same here if not normalized
    # Check max value -> if large integer, probably counts
    if verbose: print("Checking normalization...")
    if sp.issparse(adata.X):
        max_val = adata.X.data.max() if adata.X.nnz > 0 else 0
    else:
        max_val = adata.X.max()
        
    is_log = (max_val < 20) # Heuristic
    
    # Use a working copy to avoid mutating the original adata.X
    adata_calc = adata.copy()

    if not is_log:
         if verbose: print("Normalizing counts...")
         sc.pp.normalize_total(adata_calc, target_sum=1e4)
         sc.pp.log1p(adata_calc)
    
    # Check for PCA
    if 'X_pca' not in adata_calc.obsm or adata_calc.obsm['X_pca'].shape[1] < n_components:
        if verbose: print("Running PCA...")
        if use_gpu:
            # Transfer to GPU if needed (implicitly handled by rsc if installed properly)
            # But usually rsc expects adata.X to be on GPU or handles transfer
            # For simplicity, we assume standard flow or check if rsc.pp logic handles it.
            # rsc.pp.pca handles selection of HVG and PCA computation
             if 'highly_variable' not in adata_calc.var:
                 rsc.pp.highly_variable_genes(adata_calc, n_top_genes=n_features)
             rsc.pp.pca(adata_calc, n_comps=n_components)
        else:
             if 'highly_variable' not in adata_calc.var:
                 sc.pp.highly_variable_genes(adata_calc, n_top_genes=n_features)
             sc.pp.pca(adata_calc, n_comps=n_components)
             
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
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=3)
            kmeans_labels = kmeans.fit_predict(X_pca)
            kmeans_labels_cpu = kmeans_labels
            X_pca_cpu = X_pca
            
        # 2. Aggregate to centroids
        # Using pandas for fast groupby mean
        df_pca = pd.DataFrame(X_pca_cpu)
        df_pca['label'] = kmeans_labels_cpu
        centroids = df_pca.groupby('label').mean().values
        unique_labels = np.sort(np.unique(kmeans_labels_cpu))
        
        # 3. Build KNN on centroids
        if verbose: print("Building KNN graph on centroids...")
        
        # We need an AnnData object for Scanpy/RSC graph functions
        adata_centroids = AnnData(X=centroids)
        
        # R: k = min(max(2, floor(sqrt(n_centroids))-1), 10)
        n_centroids = centroids.shape[0]
        n_neighbors = min(max(2, int(np.sqrt(n_centroids)) - 1), 10)
        
        if use_gpu:
            rsc.pp.neighbors(adata_centroids, n_neighbors=n_neighbors, n_pcs=n_components, use_rep='X')
            rsc.tl.leiden(adata_centroids)
            centroid_clusters = adata_centroids.obs['leiden'].values
        else:
            sc.pp.neighbors(adata_centroids, n_neighbors=n_neighbors, n_pcs=n_components, use_rep='X')
            sc.tl.leiden(adata_centroids)
            centroid_clusters = adata_centroids.obs['leiden'].values
            
        # 4. Map back to cells
        # Create a mapping dictionary: kmeans_label -> leiden_label
        # Note: 'leiden' column is usually string/categorical
        
        # Make sure mapping aligns with unique_labels
        # If groupby sorts by label (default is True), centroids are ordered by unique_labels
        
        map_dict = dict(zip(unique_labels, centroid_clusters))
        
        # Use kmeans_labels_cpu for mapping (safe for both CPU/GPU paths)
        final_clusters = np.array([map_dict[l] for l in kmeans_labels_cpu])
        
        adata.obs[key_added] = pd.Categorical(final_clusters)
        
    else:
        # Fallback for small datasets: direct graph clustering
        if verbose: print("Running direct graph clustering (dataset small)...")
        n_neighbors = min(max(2, int(np.sqrt(n_cells)) - 1), 10)
        
        if use_gpu:
             rsc.pp.neighbors(adata_calc, n_neighbors=n_neighbors, use_rep='X_pca')
             rsc.tl.leiden(adata_calc, key_added=key_added)
        else:
             sc.pp.neighbors(adata_calc, n_neighbors=n_neighbors, use_rep='X_pca')
             sc.tl.leiden(adata_calc, key_added=key_added)
             
        adata.obs[key_added] = adata_calc.obs[key_added].values
            
    if verbose:
        n_found = len(adata.obs[key_added].unique())
        print(f"Fast clustering found {n_found} clusters.")
