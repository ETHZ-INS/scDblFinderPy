import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings

from .clustering import fast_cluster
from .doublet_generation import get_artificial_doublets
from .misc import cxds2, select_features
from .thresholding import _gdbr, doublet_thresholding_optim

def compute_doublet_score(
    adata, 
    n_neighbors=None,
    n_features=1352, 
    n_components=20,
    artificial_doublets_ratio=1.0, # Approx ratio to n_cells or fixed number? R uses fixed formula.
    n_artificial=None,
    clusters_col=None,
    samples_col=None, # R: samples=NULL
    use_gpu=False,
    random_state=42,
    n_iters=3,
    # R parameter equivalents
    clust_cor=None, # clustCor
    prop_random=0.0, # propRandom
    prop_markers=0.0, # propMarkers
    aggregate_features=False, # aggregateFeatures
    score_metric='logloss', # metric
    nrounds=0.25,
    max_depth=4,
    eta=0.3,
    filter_unidentifiable=True,
    unident_th=None,
    training_features='default', # trainingFeatures
    dbr=None,
    dbr_sd=None,
    dbr_per1k=0.008,
    stringency=0.5,
    return_type='adata', 
    verbose=True
):
    """
    Main function to compute doublet scores using the scDblFinder method.
    
    Parameters
    ----------
    adata : AnnData
        Input data.
    n_neighbors : int, optional
        Number of neighbors for KNN. If None, uses heuristic.
    n_features : int
        Number of highly variable genes to use.
    n_components : int
        Number of PCA components.
    n_artificial : int, optional
        Number of artificial doublets to generate. If None, derived from dataset size/clusters.
    clusters_col : str
        Column in adata.obs storing cluster labels. If not present, fast_cluster is run.
    samples_col : str, optional
        Column in adata.obs storing sample information. 
        If provided and multi_sample_mode is 'split', processing is done per sample.
    use_gpu : bool
        Whether to use GPU acceleration where possible.
    random_state : int
        Random seed.
    n_iters : int
        Number of iterations for the classifier training. 
    clust_cor : int or matrix, optional
        Include correlations to cell type averages. Not yet implemented.
    prop_random : float
        Proportion of artificial doublets to be made of random cells.
    aggregate_features : bool
        Whether to perform feature aggregation (for ATAC). Not yet implemented.
    score_metric : str
        Error metric for XGBoost (e.g. 'logloss').
    training_features : list or str
        Features to use for training. 'default' uses standard set.
    multi_sample_mode : str
        'split', 'singleModel', 'asOne'. Currently only 'asOne' (global) logic is detailed below.
    return_type : str
        'adata': matches input, adds scores.
        'full': returns the extended AnnData with artificial doublets.
    verbose : bool
        Print progress.
        
    Returns
    -------
    AnnData
        The input AnnData with 'scDblFinder_score' and 'scDblFinder_class' in `.obs`.
        If return_type='full', returns combined object.
    """
    
    if samples_col is not None:
        warnings.warn("multi_sample_mode has been removed; samples_col is ignored.")

    # 1. Preprocessing & Clustering
    # -----------------------------
    # Ensure counts layer or X is counts
    if 'counts' not in adata.layers:
        # Check if X is integers
        if sp.issparse(adata.X):
            is_int = np.all(np.mod(adata.X.data, 1) == 0)
        else:
            is_int = np.all(np.mod(adata.X, 1) == 0)
            
        if is_int:
            adata.layers['counts'] = adata.X.copy()
        else:
            warnings.warn("adata.X does not seem to contain raw counts and 'counts' layer is missing. Using X as is, but results may be suboptimal.")
    
    # Check normalization for later steps
    # We work with a copy for processing to avoid modifying input too much until end
    # But for clustering we might modify input
    
    if clusters_col:
        if clusters_col not in adata.obs:
            if verbose: print("Clustering cells...")
            fast_cluster(adata, n_features=n_features, n_components=n_components, 
                         key_added=clusters_col, use_gpu=use_gpu, random_state=random_state,
                         verbose=verbose)
        
        clusters = adata.obs[clusters_col].values
        n_clusters = len(np.unique(clusters))
    else:
        clusters = None
        n_clusters = 1

    if unident_th is None:
        unident_th = 0.2 if clusters is None else 0.0
        
    adata_orig = adata
    
    # Feature Selection
    # -----------------
    if n_features is not None and adata.n_vars > n_features:
        if verbose:
            print(f"Selecting top {n_features} features...")
        sel_features = select_features(
            adata, 
            clusters=clusters, 
            n_features=n_features, 
            prop_markers=prop_markers
        )
        # Keep original metadata from adata but apply subset for the rest of pipeline
        adata = adata[:, sel_features].copy()
        
    # 2. Artificial Doublet Generation
    # --------------------------------
    n_cells = adata.n_obs
    if n_artificial is None:
        # R logic: min(25000, max(1500, ceiling(ncol(sce)*0.8), 10*length(unique(cl))^2 ))
        n_artificial = min(25000, max(1500, int(n_cells * 0.8), 10 * n_clusters**2))
    
    if verbose: print(f"Generating {n_artificial} artificial doublets...")
    
    # get_artificial_doublets expects counts matrix
    X_counts = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    
    # This returns a dictionary {"counts": ..., "origins": ...}
    res = get_artificial_doublets(
        X_counts, 
        n=n_artificial, 
        clusters=clusters,
        prop_random=prop_random
    )
    X_artificial = res['counts']
    origins = res['origins']
    
    # Create AnnData for artificial
    adata_art = AnnData(X=X_artificial)
    adata_art.obs['type'] = 'artificial'
    adata_art.obs['src'] = 'artificial'
    adata_art.obs['most_likely_origin'] = origins
    
    # Prepare real adata for merge using raw counts explicitly.
    # `adata.X` may have been altered during clustering/preprocessing steps.
    X_real_counts = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    adata_real = AnnData(X=X_real_counts.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    adata_real.obs['type'] = 'real'
    adata_real.obs['src'] = 'real'
    adata_real.obs['most_likely_origin'] = np.nan # Initially unknown check if string or nan
    
    # Concatenate
    # We only care about matching genes.
    # Ensure var names match
    adata_art.var_names = adata_real.var_names
    
    # Use concat instead of deprecated concatenate
    # Note: index_unique='-' appends keys to index to ensure uniqueness
    adata_combined = sc.concat(
        [adata_real, adata_art], 
        join='outer', 
        label='batch_source', 
        keys=['real', 'artificial'], 
        index_unique='-'
    )

    # Note: concat does not preserve uns/obsm by default unless merged? 
    # But adata_real has PCA/clusters?
    # We re-run PCA anyway on combined data.
    
    # 3. Feature Calculation (Pre-PCA)
    # --------------------------------
    if verbose: print("Calculating features (CXDS, etc.)...")
    
    # CXDS
    # We calculate CXDS on the combined dataset
    # R: cxds2(e, whichDbls=which(ctype=="doublet"))
    # In R 'e' contains real+artificial. 'ctype' distinguishes them.
    # whichDbls argument tells cxds to exclude these from learning the gene pairs, but score them.
    # We exclude artificial doublets from learning.
    
    art_indices = np.where(adata_combined.obs['type'] == 'artificial')[0]
    
    scores_cxds = cxds2(adata_combined, which_dbls=art_indices, n_top=500, verbose=verbose)
    adata_combined.obs['cxds_score'] = scores_cxds
    
    # Library size & n_features
    # scanpy calculates these automatically in pp.calculate_qc_metrics usually
    if sp.issparse(adata_combined.X):
        adata_combined.obs['n_features'] = adata_combined.X.getnnz(axis=1)
        # Check if X is integer to sum?
        # Standard lib size
        adata_combined.obs['total_counts'] = np.array(adata_combined.X.sum(axis=1)).flatten()
    else:
        adata_combined.obs['n_features'] = np.count_nonzero(adata_combined.X, axis=1)
        adata_combined.obs['total_counts'] = np.sum(adata_combined.X, axis=1)
        
    # 4. Dimension Reduction (PCA)
    # ----------------------------
    if verbose: print("Processing and running PCA...")
    
    # Normalize & Log & PCA
    # We should perform this on the combined dataset
    adata_combined.layers['counts'] = adata_combined.X.copy() # Backup counts if needed
    
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)
    sc.pp.pca(adata_combined, n_comps=n_components)
    
    # 5. KNN & Doublet Features
    # -------------------------
    if verbose: print("Evaluating KNN features...")
    
    if n_neighbors is None:
         # R heuristic? default k is often based on dataset size
        n_neighbors = int(np.round(0.01 * n_cells))
        n_neighbors = max(20, min(100, n_neighbors))
        
    knn_features = _evaluate_knn(adata_combined, n_neighbors=n_neighbors, use_gpu=use_gpu)
    
    # Add features to obs
    for col, values in knn_features.items():
        adata_combined.obs[col] = values
        
    # 6. Classifier Training
    # ----------------------
    # We want to distinguish 'real' vs 'artificial'?
    # Actually, we assume 'real' are mix of singlets and doublets.
    # 'artificial' are known doublets.
    # We label:
    #   Real -> ? (mostly singlet)
    #   Artificial -> Doublet
    
    # In R implementation:
    # ctype factor: 1=real (or singlet assumption), 2=doublet (artificial + known)
    # inclInTrain: real=TRUE, artificial=TRUE.
    # Then iteratively: real cells with high score are removed from training (inclInTrain=FALSE)
    
    # Prepare training data
    # Features to use:
    # R defaults: setdiff(all, meta_cols)
    # Explicitly R excludes: distanceToNearest, distanceToNearestDoublet
    # usage: cxds_score, total_counts, n_features, weighted_density, distance_to_real, ratio_doublets, difficulty
    
    if training_features == 'default':
        feature_cols = [
            'cxds_score', 
            'total_counts', 
            'n_features', 
            'weighted_density', 
            'distance_to_real', 
            'difficulty'
        ]
        k_vals = np.sort(np.unique([k for k in [3, 10, 15, 20, 25, 50, n_neighbors] if k <= n_neighbors]))
        feature_cols.extend([f'ratio_doublets_{ki}' for ki in k_vals])
    else:
        feature_cols = training_features
        
    # remove non-numeric or non-feature keys
    feature_cols = [c for c in feature_cols if c in adata_combined.obs.columns]
    
    if verbose: print(f"Training features: {feature_cols}")
    
    # Add PCA components as features?
    # R: addVals=pca[,includePCs] -> Yes, PCA coords are used.
    # We can handle this by constructing X_train as concat of obs[features] and obsm['X_pca']
    
    # Initial labels
    # 0 = Real (Assumed Singlet), 1 = Artificial Doublet
    # Note: If we had known doublets in real data, they would be 1.
    
    y = np.zeros(adata_combined.n_obs, dtype=int)
    y[adata_combined.obs['type'] == 'artificial'] = 1
    
    # Training mask
    train_mask = np.ones(adata_combined.n_obs, dtype=bool)
    adata_combined.obs['include.in.training'] = True
    
    # Iterative Training
    # n_iters=1 means run once.
    
    scores = None

    # R-style initial score seed before iterative xgboost training.
    ratio_max = np.max(adata_combined.obs['ratio_doublets'].values)
    ratio_scaled = adata_combined.obs['ratio_doublets'].values / max(ratio_max, 1e-12)
    scores = (adata_combined.obs['cxds_score'].values + ratio_scaled) / 2.0
    adata_combined.obs['scDblFinder_score'] = scores

    # Match .scDblscore default handling when dbr.sd is not provided.
    d_tmp = adata_combined.obs[['src']].copy()
    gdbr = _gdbr(d_tmp, dbr=dbr, dbr_per1k=dbr_per1k)
    dbr_sd_eff = dbr_sd if dbr_sd is not None else (0.3 * gdbr + 0.025)

    # Build full design matrix once (features + PCA), equivalent to `preds` + `addVals`.
    X_features = adata_combined.obs[feature_cols].values
    X_pca = adata_combined.obsm['X_pca']
    X_full = np.hstack([X_features, X_pca])

    real_mask_all = adata_combined.obs['type'].values == 'real'
    dbl_mask_all = ~real_mask_all

    for i in range(n_iters):
        if verbose:
            print(f"Training iteration {i+1}/{n_iters}...")

        cols = ["type", "src", "include.in.training"]
        if clusters_col:
            cols.append(clusters_col)
        iter_df = adata_combined.obs[cols].copy()
        if clusters_col:
            iter_df = iter_df.rename(columns={clusters_col: "cluster"})
        else:
            iter_df["cluster"] = 1
        iter_df["score"] = scores

        # R equivalent: call thresholding with higher stringency (0.7) during training exclusion.
        _, calls_iter = doublet_thresholding_optim(
            iter_df,
            dbr=dbr,
            dbr_sd=dbr_sd_eff,
            dbr_per1k=dbr_per1k,
            stringency=0.7,
        )
        calls_iter = np.asarray(calls_iter)

        # w1: high-scoring real cells.
        w1_mask = real_mask_all & (calls_iter == 'doublet')
        n_real = int(np.sum(real_mask_all))
        if np.sum(w1_mask) > (n_real / 3.0):
            cap_n = int(np.floor(0.2 * n_real))
            real_idx = np.where(real_mask_all)[0]
            ord_idx = real_idx[np.argsort(-scores[real_idx])]
            keep_idx = ord_idx[:cap_n]
            w1_mask = np.zeros_like(real_mask_all, dtype=bool)
            w1_mask[keep_idx] = True

        # w2: likely unidentifiable artificial doublets.
        w2_mask = dbl_mask_all & (scores < unident_th) & bool(filter_unidentifiable)
        n_dbl = int(np.sum(dbl_mask_all))
        if bool(filter_unidentifiable) and np.sum(w2_mask) > (n_dbl / 4.0):
            cap_n = int(np.floor(0.1 * n_dbl))
            dbl_idx = np.where(dbl_mask_all)[0]
            ord_idx = dbl_idx[np.argsort(scores[dbl_idx])]
            keep_idx = ord_idx[:cap_n]
            w2_mask = np.zeros_like(dbl_mask_all, dtype=bool)
            w2_mask[keep_idx] = True

        excluded = w1_mask | w2_mask | (~train_mask)

        if verbose:
            print(f"iter={i}, {int(np.sum(excluded))} cells excluded from training.")

        train_idx = np.where(~excluded)[0]
        if train_idx.size > 10 and np.unique(y[train_idx]).size == 2:
            X_train = X_full[train_idx]
            y_train = y[train_idx]

            if nrounds is None or nrounds < 1:
                nrounds_eff = (1 + np.sum(y_train == 1)) * (0.25 if nrounds is None else float(nrounds))
                n_estimators = max(1, int(np.ceil(nrounds_eff)))
            else:
                n_estimators = int(max(1, nrounds))

            clf = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=eta,
                objective='binary:logistic',
                eval_metric=score_metric,
                subsample=0.75,
                n_jobs=-1,
                random_state=random_state,
            )
            clf.fit(X_train, y_train)
            scores = clf.predict_proba(X_full)[:, 1]

        adata_combined.obs['scDblFinder_score'] = scores

        # Update difficulty proxy from model score similarly to R iterative update.
        if 'most_likely_origin' in adata_combined.obs.columns:
            origin_series = adata_combined.obs['most_likely_origin']
            valid_origin = origin_series.notna().values
            non_real_valid = dbl_mask_all & valid_origin
            if np.any(non_real_valid):
                class_diff = (
                    pd.DataFrame({'origin': origin_series[non_real_valid].values, 'score': scores[non_real_valid]})
                    .groupby('origin', sort=False)['score']
                    .mean()
                )
                default_diff = float(class_diff.mean())
                difficulty = np.full(adata_combined.n_obs, default_diff, dtype=float)
                mapped = class_diff.reindex(origin_series.values)
                mapped_vals = mapped.to_numpy(dtype=float)
                m = ~np.isnan(mapped_vals)
                difficulty[m] = 1.0 - mapped_vals[m]
                adata_combined.obs['difficulty'] = difficulty

        train_mask = ~excluded
        adata_combined.obs['include.in.training'] = train_mask

    # 7. Thresholding (optim)
    # -----------------------
    # Keep parity with the R `optim` threshold path used in the current flow.
    cols = ["type", "src", "include.in.training"]
    if clusters_col:
        cols.append(clusters_col)
    threshold_df = adata_combined.obs[cols].copy()
    if clusters_col:
        threshold_df = threshold_df.rename(columns={clusters_col: "cluster"})
    else:
        threshold_df["cluster"] = 1
    threshold_df["score"] = scores
    final_threshold, final_calls = doublet_thresholding_optim(
        threshold_df,
        dbr=dbr,
        dbr_sd=dbr_sd_eff,
        dbr_per1k=dbr_per1k,
        stringency=stringency,
    )
    adata_combined.obs['scDblFinder_threshold'] = final_threshold
    adata_combined.obs['scDblFinder_class'] = pd.Categorical(final_calls)
    
    # Let's map scores back to original `adata`
    # We only care about 'real' cells
    
    real_mask = adata_combined.obs['type'] == 'real'
    final_scores = scores[real_mask]
    
    # Store in original adata
    # Assuming order preserved? 
    # adata_real was first part of concat.
    # Check simple length match
    if len(final_scores) != n_cells:
        # Fallback to index matching
        # adata.obs_names should be in adata_combined.obs_names (maybe with suffixes)
        # Actually standard concat appends if keys match.
        pass
        
    adata_orig.obs['scDblFinder_score'] = final_scores
    adata_orig.obs['scDblFinder_class'] = pd.Categorical(np.asarray(final_calls)[real_mask])
    adata_orig.uns['scDblFinder_threshold'] = float(final_threshold)
    
    if return_type == 'full':
        # Add artificial to orig
        adata_all = sc.concat([adata_orig, adata_combined[~real_mask].copy()], join='outer')
        return adata_all
    
    return adata_orig


def _evaluate_knn(adata, n_neighbors=50, use_gpu=False):
    """
    Calculates KNN-based features for doublet detection.
    
    Features:
    - distance_to_nearest (distance to kth neighbor)
    - weighted_doublet_density
    - ratio_doublets_k (ratio of artificial doublets in neighborhood)
    - difficulty (based on most likely origin)
    """
    X_pca = adata.obsm['X_pca']
    y_type = (adata.obs['type'] == 'artificial').values.astype(int) # 0=Real, 1=Artificial
    
    # Origins tracking
    # Get origins from obs. 
    # Artificial cells have known origin (e.g. 'ClusterA+ClusterB'). Real cells have NaN.
    
    # We need to map string origins to integers for efficient processing or use arrays?
    origins = adata.obs['most_likely_origin'].values.copy()
    
    # Train KNN
    # Using sklearn for consistency, or scanpy
    # Match R/BiocNeighbors behavior by excluding each point itself from neighbors.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto', n_jobs=-1).fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    # Drop self-neighbor column (typically first, with distance 0 and identical index).
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    n_obs = X_obs = X_pca.shape[0]
    
    # 1. Distance to nearest (kth)
    # distances is (n_obs, n_neighbors). Last column is distance to kth.
    dist_to_k = distances[:, -1]
    
    # 2. Ratio of doublets in neighborhood
    # indices is (n_obs, n_neighbors)
    # Retrieve types of neighbors
    neighbor_types = y_type[indices] # (n_obs, n_neighbors)
    
    # Ratio at multiple k
    ratio_dict = {}
    k_vals = np.sort(np.unique([k for k in [3, 10, 15, 20, 25, 50, n_neighbors] if k <= n_neighbors]))
    for ki in k_vals:
        ratio_dict[f'ratio_doublets_{ki}'] = np.mean(neighbor_types[:, :ki], axis=1)
    
    # 3. Weighted density
    # R: dw <- sqrt(k - seq_len(k)) * 1/dist
    # Python: weights based on rank/distance
    
    # Check for zero dists
    SAFE_DIST = distances.copy()
    first_col = SAFE_DIST[:, 0]
    min_gt_0 = first_col[first_col > 0].min() if (first_col > 0).any() else 1e-6
    SAFE_DIST[SAFE_DIST == 0] = min_gt_0
    
    ranks = np.arange(1, n_neighbors + 1)
    rank_weights = np.sqrt(n_neighbors - ranks) # Shape (n_neighbors,)
    
    # Distance weighting: 1 / distance
    dist_weights = 1.0 / SAFE_DIST
    
    # Combined weights
    weights = rank_weights * dist_weights
    
    # Normalize rows
    row_sums = weights.sum(axis=1, keepdims=True)
    norm_weights = weights / row_sums
    
    weighted_score = np.sum(neighbor_types * norm_weights, axis=1)

    # 4. Most Likely Origin & Difficulty
    # R: origins determined by looking at neighbors' origins.
    # Real cells get origin assigned based on frequent neighbors.
    
    # Retrieve neighbor origins
    # neighbor_origins (N x k). Contains strings or NaNs.
    neighbor_origins = origins[indices]
    
    # For each cell, determine most frequent origin among neighbors
    # Ignore NaNs (real neighbors)
    
    most_likely = []
    
    # This loop is slow in Python for large N. Optimize?
    # Vectorized approach hard with strings.
    # Map unique origins to ints.
    
    unique_origins = pd.unique(origins[~pd.isnull(origins)])
    origin_map = {o: i for i, o in enumerate(unique_origins)}
    rev_origin_map = {i: o for i, o in enumerate(unique_origins)}
    n_origins = len(unique_origins)
    
    if n_origins > 0:
        # Convert origins to numeric, -1 for NaN
        origins_num = np.full(len(origins), -1, dtype=int)
        valid_mask = ~pd.isnull(origins)
        # Use pandas map is faster?
        # origins_num[valid_mask] = [origin_map[o] for o in origins[valid_mask]] # List comp slow
        # Series map
        
        s_origins = pd.Series(origins)
        # Map known
        mapped = s_origins.map(origin_map).fillna(-1).astype(int).values
        origins_num = mapped
        
        neighbor_origins_num = origins_num[indices] # (N x k)
        
        # Calculate mode per row, ignoring -1
        # Bincount per row? Too slow.
        # Scipy mode? `scipy.stats.mode` handles axis.
        
        from scipy.stats import mode
        # mode returns smallest value if multiple. -1 is smallest.
        # We want to ignore -1.
        
        # Helper to compute mode ignoring -1
        def mode_ignoring_neg1(arr):
             # Expects 2D array
             # Replace -1 with max+1 to push to end if using sort?
             # Or use bincount on flattened and reshape?
             pass

        # Simple python loop for now to be safe and correct
        # Or faster: only where ratio_k > 0 (has doublet neighbors)
        
        final_origins = np.full(n_obs, -1, dtype=int)
        
        # We can just iterate. 10k cells x 50 neighbors is 500k ops, fast enough.
        # Actually standard python loop is slow.
        
        # Use simple heuristic: if ratio_doublets > 0, likely has origin.
        # Most frequent positive integer in row.
        
        # Optimization: use pandas apply on the matrix of neighbor indices? No.
        
        for i in range(n_obs):
            row = neighbor_origins_num[i]
            valid_neighbors = row[row >= 0]
            if len(valid_neighbors) > 0:
                # Find mode
                vals, counts = np.unique(valid_neighbors, return_counts=True)
                final_origins[i] = vals[np.argmax(counts)]
                
        # String origins
        most_likely_str = np.array([rev_origin_map[i] if i >= 0 else np.nan for i in final_origins], dtype=object)
        
    else:
        most_likely_str = np.full(n_obs, np.nan, dtype=object)
        
    # Difficulty Feature
    # R: class.weighted <- mean(weighted[type=="doublet"]) per origin
    # D$difficulty[w] <- 1 - class.weighted[origin]
    
    difficulty = np.ones(n_obs, dtype=float)
    
    if n_origins > 0:
        # Compute mean weighted score per origin (using only artificial doublets)
        df = pd.DataFrame({'origin': origins, 'weighted': weighted_score, 'type': y_type})
        
        # Filter for artificial doublets
        df_art = df[df['type'] == 1]
        
        # Groupby origin
        origin_means = df_art.groupby('origin')['weighted'].mean()
        
        # Map means to all cells based on most_likely_str
        # If most_likely_str is NaN, difficulty remains 1? 
        # R: d$difficulty <- 1; d$difficulty[w] <- 1 - class.weighted...
        
        # Map
        mapped_means = origin_means.reindex(most_likely_str).values
        
        # Where mapped_means is valid (not NaN), update difficulty
        valid_means = ~np.isnan(mapped_means)
        difficulty[valid_means] = 1.0 - mapped_means[valid_means]

    
    # 5. Dist to nearest Real
    # Efficiently find min dist to type 0
    
    real_mask = (neighbor_types == 0)
    max_dist = distances[:, 0].max() * 2
    d_real = distances.copy()
    d_real[~real_mask] = max_dist
    dist_to_nearest_real = d_real.min(axis=1)

    res = {
        'distance_to_nearest': dist_to_k, # Keep for debug but exclude from features later
        'weighted_density': weighted_score,
        'distance_to_real': dist_to_nearest_real,
        'difficulty': difficulty,
        'most_likely_origin': most_likely_str,
        'ratio_doublets': np.mean(neighbor_types[:, :n_neighbors], axis=1)
    }
    res.update(ratio_dict)
    return res
