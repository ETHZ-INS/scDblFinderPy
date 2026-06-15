import numpy as np
import pandas as pd
import scanpy as sc
import random
import os
import json
import time
from anndata import AnnData
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings
import glob

from .clustering import fast_cluster
from .doublet_generation import get_artificial_doublets
from .misc import cxds2, select_features
from .thresholding import _gdbr, doublet_thresholding_optim
from .rng import coerce_rng

def _filter_unrecognizable_doublets(d, minSize=5, minMedDiff=0.1):
    src = d['src'].values
    origin = d['most_likely_origin'].values if 'most_likely_origin' in d else np.full(len(d), np.nan)
    score = d['score'].values
    cluster = d['cluster'].values
    
    da_mask = (src == 'artificial') & pd.Series(origin).str.contains(r'\+', regex=True, na=False).values
    dr_mask = (src == 'real')
    
    if not np.any(dr_mask):
        return []
        
    dr_med = np.median(score[dr_mask])
    
    dr_df = pd.DataFrame({'score': score[dr_mask], 'cluster': cluster[dr_mask]})
    if dr_df.empty:
        return []
    
    rq = dr_df.groupby('cluster', observed=False)['score'].quantile([0.5, 0.9]).unstack()
    
    da_df = pd.DataFrame({'score': score[da_mask], 'origin': origin[da_mask]})
    da_groups = da_df.groupby('origin', observed=False)
    
    drop_origins = []
    for orig, group in da_groups:
        if len(group) < minSize:
            continue
        z = group['score'].quantile([0.1, 0.5]).values
        origs = str(orig).split('+')
        if origs[0] not in rq.index or origs[1] not in rq.index:
            continue
            
        rq_x_2 = [rq.loc[origs[0] , 0.9], rq.loc[origs[1], 0.9]]
        rq_x_1_max = max(rq.loc[origs[0] , 0.5], rq.loc[origs[1], 0.5])
        
        if any(z[0] < v for v in rq_x_2) or (z[1] - max(dr_med, rq_x_1_max)) < minMedDiff:
            drop_origins.append(orig)
            
    return drop_origins

def compute_doublet_score(
    adata, 
    n_neighbors=None,
    n_features=1000, 
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
    prop_random=0.1, # propRandom (R default)
    adjust_size=0.25,
    meta_triplets=True,
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
    verbose=True,
    debug=False
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

    # Central RNG: drives all randomness in the pipeline
    central_rng = coerce_rng(random_state=random_state)

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
            # derive an int seed for external libraries from central RNG
            fast_cluster(adata, n_features=n_features, n_components=n_components, 
                         key_added=clusters_col, use_gpu=use_gpu, rng=central_rng,
                         verbose=verbose)
        
        clusters = adata.obs[clusters_col].values
        n_clusters = len(np.unique(clusters))
    else:
        clusters = None
        n_clusters = 1

    if unident_th is None:
        unident_th = 0.0 if clusters is not None else 0.2
        
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
        n_artificial = min(25000, max(1500, int(np.ceil(n_cells * 0.8)), 10 * n_clusters**2))
    
    if verbose: print(f"Generating {n_artificial} artificial doublets...")
    
    # get_artificial_doublets expects counts matrix
    X_counts = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    
    # This returns a dictionary {"counts": ..., "origins": ...}
    # Pass central RNG into artificial doublet generator so all stochastic
    # operations derive from the same sequence.
    res = get_artificial_doublets(
        X_counts, 
        n=n_artificial, 
        clusters=clusters,
        prop_random=prop_random,
        adjust_size=adjust_size,
        meta_triplets=meta_triplets,
        random_state=random_state,
        rng=central_rng
    )
    X_artificial = res['counts']
    origins = res['origins']
    art_types = res.get('types', ['art'] * len(origins))

    # Create AnnData for artificial
    adata_art = AnnData(X=X_artificial)
    # Name artificial cells: random doublets get "rDbl." prefix to match R's createDoublets
    # prefix="rDbl." convention. .optimThreshold checks row names for "^rDbl\." to estimate
    # expected false negatives from homotypic doublets.
    art_names = [
        f"rDbl.{i+1}" if t == 'rDbl' else f"art.{i+1}"
        for i, t in enumerate(art_types)
    ]
    try:
        adata_art.obs_names = art_names
    except Exception:
        # fallback: set a column instead
        adata_art.obs['__art_name'] = art_names
    adata_art.obs['type'] = 'artificial'
    adata_art.obs['src'] = 'artificial'
    adata_art.obs['most_likely_origin'] = origins
    
    # Prepare real adata for merge using raw counts explicitly.
    # `adata.X` may have been altered during clustering/preprocessing steps.
    X_real_counts = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    adata_real = AnnData(X=X_real_counts.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    adata_real.obs['type'] = 'real'
    adata_real.obs['src'] = 'real'
    adata_real.obs['most_likely_origin'] = np.nan # Initially unknown
    # Preserve original obs names to ensure correct mapping after concat
    adata_real.obs['_orig_index'] = adata_real.obs_names.astype(str)
    # Mark artificial originals as NaN so we can distinguish
    adata_art.obs['_orig_index'] = [np.nan] * adata_art.n_obs
    
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
    
    # Normalize & PCA - align with R's .defaultProcessing: normalizeCounts then PCA via Irlba
    # Use the raw counts backup for R-like normalization
    adata_combined.layers['counts'] = adata_combined.X.copy()
    counts_mat = adata_combined.layers['counts']
    # ensure dense numeric array for SVD
    if sp.issparse(counts_mat):
        counts_arr = counts_mat.toarray().astype(float)
    else:
        counts_arr = np.asarray(counts_mat, dtype=float)

    # R normalizeCounts: library-size normalize, then transform before PCA.
    lib_sizes = counts_arr.sum(axis=1)
    mean_lib = lib_sizes.mean() if lib_sizes.size>0 else 1.0
    size_factors = lib_sizes / mean_lib
    # avoid division by zero
    size_factors[size_factors == 0] = 1.0
    norm_counts = counts_arr / size_factors[:, None]
    # scuttle::normalizeCounts stores log-normalized values; use log2(1+x) to
    # match the Bioconductor convention more closely than scanpy's default log1p.
    norm_counts = np.log2(norm_counts + 1.0)

    # center genes (columns) as scater::calculatePCA does centering before SVD
    gene_means = np.mean(norm_counts, axis=0)
    norm_centered = norm_counts - gene_means

    # Compute PCA in pure Python using randomized SVD to mimic R's irlba
    try:
        from sklearn.utils.extmath import randomized_svd
        n_comp_eff = min(n_components, min(norm_centered.shape))
        # randomized_svd expects (n_samples, n_features) array
        # use a deterministic seed derived from random_state for reproducibility
        rand_state = random_state if isinstance(random_state, (int, float)) else 42
        U, S, VT = randomized_svd(norm_centered, n_components=n_comp_eff, random_state=int(rand_state))
        pcs = U[:, :n_comp_eff] * S[:n_comp_eff]
    except Exception:
        try:
            # fallback to full SVD
            U, S, VT = np.linalg.svd(norm_centered, full_matrices=False)
            k = min(n_components, U.shape[1])
            pcs = U[:, :k] * S[:k]
        except Exception:
            # final fallback: use sklearn PCA on the log-normalized counts
            from sklearn.decomposition import PCA
            n_comp_eff = min(n_components, min(norm_centered.shape))
            pca = PCA(n_components=n_comp_eff, svd_solver='auto', random_state=int(random_state if isinstance(random_state, (int, float)) else 42))
            pcs = pca.fit_transform(norm_centered)

    # store PCA coordinates in obsm as cells x components
    adata_combined.obsm['X_pca'] = pcs

    # Compute cluster-correlations predictors if requested (clustCor in R)
    # R computes these on the counts matrix before PCA and adds them to the
    # predictors table. Here we approximate that behavior: when `clust_cor` is
    # an integer, pick top-n markers per cluster by mean expression; when it is
    # a 2D array/DataFrame, use its columns as signatures and correlate.
    if clust_cor is not None and clusters is not None:
        try:
            from scipy.stats import rankdata
            counts_mat = adata_combined.layers['counts'] if 'counts' in adata_combined.layers else adata_combined.X
            # ensure dense for computations
            if sp.issparse(counts_mat):
                counts_dense = counts_mat.toarray()
            else:
                counts_dense = np.asarray(counts_mat)

            cluster_labels = np.asarray(clusters)
            uniq_clusters = np.unique(cluster_labels)
            clustcor_cols = []
            if hasattr(clust_cor, 'shape') and len(np.shape(clust_cor)) == 2:
                # matrix-like: rows should correspond to genes (var_names)
                if isinstance(clust_cor, pd.DataFrame):
                    mat = clust_cor.copy()
                else:
                    mat = pd.DataFrame(clust_cor, index=adata_combined.var_names)
                # intersect genes
                common = np.intersect1d(adata_combined.var_names, mat.index)
                if len(common) >= 5:
                    sub_counts = pd.DataFrame(counts_dense[:, [list(adata_combined.var_names).index(g) for g in common]], columns=common)
                    for col in mat.columns:
                        sig = mat.loc[common, col].values
                        # compute Spearman correlation between each cell and signature
                        # compute ranks
                        sig_rank = rankdata(sig)
                        vals = []
                        for r in range(sub_counts.shape[0]):
                            cell_rank = rankdata(sub_counts.iloc[r].values)
                            # Pearson on ranks
                            num = np.cov(cell_rank, sig_rank, bias=True)[0,1]
                            den = np.std(cell_rank) * np.std(sig_rank)
                            vals.append(0.0 if den == 0 else num/den)
                        colname = f'clustCor_{col}'
                        adata_combined.obs[colname] = vals
                        clustcor_cols.append(colname)
            else:
                # integer: pick markers per cluster
                try:
                    nmark = int(clust_cor)
                except Exception:
                    nmark = None
                if nmark and nmark > 0:
                    # compute cluster means (cells x genes)
                    means = {}
                    markers = set()
                    for cl in uniq_clusters:
                        idx = np.where(cluster_labels == cl)[0]
                        if idx.size == 0:
                            continue
                        meanvec = counts_dense[idx].mean(axis=0)
                        top_idx = np.argsort(meanvec)[-nmark:]
                        for ti in top_idx:
                            markers.add(ti)
                        means[cl] = meanvec
                    markers = sorted(list(markers))
                    if len(markers) >= 5:
                        sub_counts = counts_dense[:, markers]
                        for cl in uniq_clusters:
                            sig = means.get(cl, np.zeros(counts_dense.shape[1]))[markers]
                            sig_rank = rankdata(sig)
                            vals = []
                            for r in range(sub_counts.shape[0]):
                                cell_rank = rankdata(sub_counts[r])
                                num = np.cov(cell_rank, sig_rank, bias=True)[0,1]
                                den = np.std(cell_rank) * np.std(sig_rank)
                                vals.append(0.0 if den == 0 else num/den)
                            colname = f'clustCor_{cl}'
                            adata_combined.obs[colname] = vals
                            clustcor_cols.append(colname)
            if verbose and len(clustcor_cols)>0:
                print(f"Added clustCor predictors: {clustcor_cols}")
        except Exception as e:
            if verbose:
                print('Failed to compute clustCor predictors:', e)
    
    # 5. KNN & Doublet Features
    # -------------------------
    if verbose: print("Evaluating KNN features...")
    
    if n_neighbors is None:
        # Align with R's defaultKnnKs kmax heuristic.
        n_neighbors = max(int(np.ceil(np.sqrt(n_cells / 2.0))), 25)
        
    knn_features = _evaluate_knn(adata_combined, n_neighbors=n_neighbors,
                                 use_gpu=use_gpu, random_state=random_state,
                                 random_mode=(clusters is None))
    
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

    # Initial labels
    # 0 = Real (Assumed Singlet), 1 = Artificial Doublet
    # Note: If we had known doublets in real data, they would be 1.
    
    y = np.zeros(adata_combined.n_obs, dtype=int)
    y[adata_combined.obs['type'] == 'artificial'] = 1
    
    # Training mask mirrors the R flow and is updated after each iteration.
    train_mask = np.ones(adata_combined.n_obs, dtype=bool)
    adata_combined.obs['include.in.training'] = train_mask
    
    # Iterative Training
    # n_iters=1 means run once.
    
    scores = None

    # R-style initial score seed before iterative xgboost training.
    ratio_feature_cols = sorted(
        [c for c in adata_combined.obs.columns if c.startswith('ratio_doublets_')],
        key=lambda x: int(x.rsplit('_', 1)[1])
    )
    if not ratio_feature_cols:
        raise ValueError('No ratio_doublets_* features were generated for the initial score')
    ratio_feature = ratio_feature_cols[-1]
    ratio_vals = adata_combined.obs[ratio_feature].values
    cxds_vals = adata_combined.obs['cxds_score'].values

    scores = (cxds_vals + ratio_vals / max(float(np.max(ratio_vals)), 1e-12)) / 2.0
    adata_combined.obs['scDblFinder_score'] = scores

    # Match .scDblscore default handling when dbr.sd is not provided.
    d_tmp = adata_combined.obs[['src']].copy()
    gdbr = _gdbr(d_tmp, dbr=dbr, dbr_per1k=dbr_per1k)
    dbr_sd_eff = dbr_sd if dbr_sd is not None else (0.3 * gdbr + 0.025)

    # Prepare training data after KNN-derived features have been added.
    # This mirrors the R flow, where the scorer sees the full post-KNN table.
    if training_features == 'default':
        excluded_features = {
            'most_likely_origin',
            'originAmbiguous',
            'distance_to_nearest_doublet',
            'distance_to_nearest',
            'scDblFinder_score',
            'type',
            'src',
            'class',
            'nearestClass',
            'cluster',
            'sample',
            'expected',
            'include.in.training',
            'observed',
        }
        ratio_cols = [c for c in adata_combined.obs.columns if c.startswith('ratio_doublets_')]
        ordered_candidates = [
            'weighted_density',
            'distance_to_nearest_real',
            *ratio_cols,
            # R never creates a 'difficulty' column when clusters=NULL (random mode),
            # so exclude it here to match that behaviour. In clustered mode it is
            # computed per-origin and is genuinely informative.
            *(['difficulty'] if clusters is not None else []),
            'total_counts',
            'n_features',
            'nAbove2',
            'cxds_score',
        ]
        # include any clustCor-derived columns (added earlier) so they are available for training
        clustcor_cols = [c for c in adata_combined.obs.columns if str(c).startswith('clustCor_')]
        if clustcor_cols:
            ordered_candidates = ordered_candidates + clustcor_cols
        feature_cols = [
            c for c in ordered_candidates
            if c in adata_combined.obs.columns
            and c not in excluded_features
            and pd.api.types.is_numeric_dtype(adata_combined.obs[c])
        ]
    else:
        feature_cols = [c for c in training_features if c in adata_combined.obs.columns]

    if verbose:
        print(f"Training features: {feature_cols}")

    # Build full design matrix once (features + PCA), equivalent to `preds` + `addVals`.
    # R's scDblFinder defaults to includePCs=19 and then keeps only PCs with index
    # strictly less than ncol(pca), so with 20 PCs it uses PC1..PC19.
    X_features = adata_combined.obs[feature_cols].values
    X_pca = adata_combined.obsm['X_pca']
    n_include_pcs = max(0, min(19, X_pca.shape[1] - 1))
    X_pca_used = X_pca[:, :n_include_pcs]
    X_full = np.hstack([X_features, X_pca_used])

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
        n_estimators_local = None
        if train_idx.size > 10 and np.unique(y[train_idx]).size == 2:
            X_train = X_full[train_idx]
            y_train = y[train_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            params = {
                'objective': 'binary:logistic',
                'eval_metric': score_metric,
                'max_depth': max_depth,
                'learning_rate': eta,
                'subsample': 0.75,
                'tree_method': 'exact',
                'nthread': 1,
                'seed': random_state,
            }

            if nrounds is None or nrounds < 1:
                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=200,
                    nfold=5,
                    early_stopping_rounds=2,
                    metrics={score_metric},
                    seed=random_state,
                    verbose_eval=False
                )
                mean_col = f"test-{score_metric}-mean"
                std_col = f"test-{score_metric}-std"
                
                best_idx = cv_results[mean_col].idxmin()
                
                if nrounds == 0 or nrounds is None:
                    n_estimators = int(max(1, best_idx + 1))
                else:
                    ac = cv_results.loc[best_idx, mean_col] + float(nrounds) * cv_results.loc[best_idx, std_col]
                    # R finds the `min(which(mean <= ac))`, meaning the first round that drops below the threshold 'ac'
                    valid_rounds = cv_results[cv_results[mean_col] <= ac].index
                    n_estimators = int(max(1, valid_rounds[0] + 1)) if len(valid_rounds) > 0 else int(max(1, best_idx + 1))
            else:
                n_estimators = int(max(1, nrounds))
            # Record chosen estimator count for debug traces
            n_estimators_local = int(n_estimators)
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators
            )
            dfull = xgb.DMatrix(X_full)
            scores = bst.predict(dfull)

            trace_bundle_ts = int(time.time()) if debug else None

            # If debugging, capture CV results, chosen rounds and model info
            if debug:
                try:
                    trace_dir = os.path.join(os.getcwd(), 'scdbl_traces')
                    os.makedirs(trace_dir, exist_ok=True)
                    trace_ts = trace_bundle_ts if trace_bundle_ts is not None else int(time.time())
                    try:
                        pca_dims = X_pca_used.shape[1]
                    except Exception:
                        pca_dims = 0
                    feature_names = list(feature_cols) + [f'PC{i+1}' for i in range(pca_dims)]
                    trace_npz = os.path.join(trace_dir, f"trace_iter_{trace_ts}_{i}.npz")
                    obs_names_arr = np.asarray(adata_combined.obs_names.astype(str)) if hasattr(adata_combined, 'obs_names') else None
                    knn_inds = None
                    knn_dists = None
                    try:
                        knn_inds = adata_combined.obsm.get('knn_indices', None)
                        knn_dists = adata_combined.obsm.get('knn_distances', None)
                    except Exception:
                        pass
                    np.savez_compressed(
                        trace_npz,
                        X_full=X_full,
                        scores=np.asarray(scores),
                        y=np.asarray(y),
                        train_idx=np.asarray(train_idx),
                        obs_names=obs_names_arr,
                        knn_indices=knn_inds,
                        knn_distances=knn_dists,
                    )
                    fn_path = os.path.join(trace_dir, f"feature_names_iter_{trace_ts}_{i}.json")
                    with open(fn_path, 'w') as ffn:
                        json.dump(feature_names, ffn)
                    if 'cv_results' in locals() and cv_results is not None:
                        cv_path = os.path.join(trace_dir, f"cv_iter_{trace_ts}_{i}.csv")
                        try:
                            cv_results.to_csv(cv_path, index=True)
                        except Exception:
                            try:
                                cv_results.to_json(cv_path + '.json', orient='split')
                            except Exception:
                                pass
                    try:
                        imp = bst.get_score(importance_type='weight')
                        imp_path = os.path.join(trace_dir, f"imp_iter_{trace_ts}_{i}.json")
                        with open(imp_path, 'w') as fimp:
                            json.dump(imp, fimp)
                    except Exception:
                        pass
                except Exception as e:
                    if verbose:
                        print(f"Failed to write debug traces: {e}")

        adata_combined.obs['scDblFinder_score'] = scores

        train_mask = ~excluded
        # Do not write `include.in.training` back to `adata_combined.obs` here.
        # R sets `d$include.in.training[w] <- FALSE` only after the iterative loop,
        # so keep `train_mask` in memory and apply it once after the loop.

        if debug:
            debug_record = {
                'timestamp': time.time(),
                'n_cells_total': int(adata_combined.n_obs),
                'iteration': int(i),
                'n_real': int(n_real),
                'n_artificial': int(n_dbl),
                'w1_count': int(np.sum(w1_mask)),
                'w2_count': int(np.sum(w2_mask)),
                'excluded_count': int(np.sum(excluded)),
                'train_idx_size': int(train_idx.size),
                'n_estimators': n_estimators_local,
                'drop_origins': list(drop_origins) if 'drop_origins' in locals() else [],
                'unident_th': float(unident_th),
                'median_score_real': float(np.median(scores[real_mask_all])) if np.any(real_mask_all) else None,
                'median_score_artificial': float(np.median(scores[dbl_mask_all])) if np.any(dbl_mask_all) else None,
            }
            try:
                debug_ts = trace_bundle_ts if trace_bundle_ts is not None else int(time.time())
                log_path = os.path.join(os.getcwd(), f"scdbl_debug_{debug_ts}.jsonl")
                with open(log_path, 'a') as f:
                    f.write(json.dumps(debug_record) + "\n")
                if verbose:
                    print(f"Wrote debug record to {log_path}")
                # Always dump the iter_df and calls_iter for parity inspection
                try:
                    trace_dir = os.path.join(os.getcwd(), 'scdbl_traces')
                    os.makedirs(trace_dir, exist_ok=True)
                    iter_csv = os.path.join(trace_dir, f"iter_df_{debug_ts}_{i}.csv")
                    # include cell identifiers as the CSV index for alignment with R
                    try:
                        iter_df.to_csv(iter_csv, index=True, index_label='cell_id')
                    except Exception:
                        iter_df.to_csv(iter_csv, index=False)
                    calls_path = os.path.join(trace_dir, f"calls_iter_{debug_ts}_{i}.npy")
                    np.save(calls_path, calls_iter)
                except Exception as e:
                    if verbose:
                        print(f"Failed to write iter_df/calls_iter: {e}")
            except Exception as e:
                if verbose:
                    print(f"Failed to write debug record: {e}")
                # Also dump the iter_df and calls_iter for parity inspection
                try:
                    debug_ts = trace_bundle_ts if trace_bundle_ts is not None else int(time.time())
                    trace_dir = os.path.join(os.getcwd(), 'scdbl_traces')
                    os.makedirs(trace_dir, exist_ok=True)
                    iter_csv = os.path.join(trace_dir, f"iter_df_{debug_ts}_{i}.csv")
                    # include cell identifiers as the CSV index for alignment with R
                    try:
                        iter_df.to_csv(iter_csv, index=True, index_label='cell_id')
                    except Exception:
                        iter_df.to_csv(iter_csv, index=False)
                    calls_path = os.path.join(trace_dir, f"calls_iter_{debug_ts}_{i}.npy")
                    # save calls_iter as array of strings
                    np.save(calls_path, calls_iter)
                except Exception as e:
                    if verbose:
                        print(f"Failed to write iter_df/calls_iter: {e}")

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

        if bool(filter_unidentifiable) and i == n_iters - 1:
            tmp_d = adata_combined.obs[['src', 'most_likely_origin', 'type']].copy()
            tmp_d['score'] = scores
            tmp_d['cluster'] = adata_combined.obs[clusters_col].values if clusters_col else 1
            drop_origins = _filter_unrecognizable_doublets(tmp_d)
            if len(drop_origins) > 0:
                drop_mask = ((tmp_d['src'] == 'artificial') & tmp_d['most_likely_origin'].isin(drop_origins)).to_numpy()
                if np.any(drop_mask):
                    keep_mask = np.asarray(~drop_mask, dtype=bool)
                    if keep_mask.shape[0] != adata_combined.n_obs:
                        raise ValueError(
                            f"Final unidentifiable-doublet mask has length {keep_mask.shape[0]} "
                            f"but adata_combined has {adata_combined.n_obs} rows"
                        )
                    adata_combined = adata_combined[keep_mask].copy()
                    scores = np.asarray(scores)[keep_mask]
                    train_mask = np.asarray(train_mask, dtype=bool)[keep_mask]
                    real_mask_all = np.asarray(real_mask_all, dtype=bool)[keep_mask]
                    dbl_mask_all = np.asarray(dbl_mask_all, dtype=bool)[keep_mask]
                    X_full = X_full[keep_mask]
                    y = np.asarray(y, dtype=int)[keep_mask]

    # 7. Thresholding (optim)
    # -----------------------
    # Apply final include.in.training flags now (matches R: set after loop)
    try:
        adata_combined.obs['include.in.training'] = train_mask
    except Exception:
        # If train_mask shape changed due to final filtering, align by length
        adata_combined.obs['include.in.training'] = np.asarray(train_mask, dtype=bool)
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

    # Map final scores back to original AnnData using preserved _orig_index
    orig_indices = adata_combined.obs.loc[real_mask, '_orig_index'].astype(str).values
    # Build series indexed by original obs names
    ser_scores = pd.Series(final_scores, index=orig_indices)
    # Reindex to the original adata ordering (will insert NaN if mismatch)
    adata_orig.obs['scDblFinder_score'] = ser_scores.reindex(adata_orig.obs_names).values

    # Map classes similarly
    final_calls_all = np.asarray(final_calls)
    final_calls_real = final_calls_all[real_mask]
    ser_calls = pd.Series(final_calls_real, index=orig_indices)
    adata_orig.obs['scDblFinder_class'] = pd.Categorical(ser_calls.reindex(adata_orig.obs_names).values)

    adata_orig.uns['scDblFinder_threshold'] = float(final_threshold)
    
    if return_type == 'full':
        # Add artificial to orig
        adata_all = sc.concat([adata_orig, adata_combined[~real_mask].copy()], join='outer')
        return adata_all
    
    return adata_orig


def _evaluate_knn(adata, n_neighbors=50, use_gpu=False, random_state=42, random_mode=False):
    """
    Calculates KNN-based features for doublet detection.

    Mirrors R's .evaluateKNN: all features are computed on k TRUE non-self
    neighbors, matching BiocNeighbors::findKNN which excludes the query point.
    """
    X_pca = adata.obsm['X_pca']
    y_type = (adata.obs['type'] == 'artificial').values.astype(int)  # 0=real, 1=artificial
    origins = adata.obs['most_likely_origin'].values.copy()

    n_obs = X_pca.shape[0]

    # Request n_neighbors + 1 so that after stripping the query point (which both
    # sklearn and pynndescent include at distance 0) we are left with exactly
    # n_neighbors true non-self neighbors — matching R's findKNN behaviour.
    _n_query = n_neighbors + 1

    # In random mode, use exact KNN for datasets up to 50k combined cells.
    # pynndescent's approximation degrades on larger datasets (e.g. hm-12k at 21k cells),
    # causing a growing AUPRC gap vs R. Exact KNN eliminates that error.
    # In cluster mode keep pynndescent: exact KNN changes cluster-doublet feature values
    # in ways that hurt AUPRC on within-tissue datasets (e.g. pbmc).
    _EXACT_KNN_THRESHOLD = 50000

    if random_mode and n_obs <= _EXACT_KNN_THRESHOLD:
        nbrs = NearestNeighbors(n_neighbors=_n_query, algorithm='ball_tree',
                                metric='euclidean', n_jobs=-1)
        nbrs.fit(X_pca)
        distances, indices = nbrs.kneighbors(X_pca)
    else:
        _knn_done = False
        try:
            import warnings as _warnings
            from pynndescent import NNDescent
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _nn_index = NNDescent(X_pca, n_neighbors=_n_query,
                                      metric='euclidean', random_state=random_state)
            indices, distances = _nn_index.neighbor_graph
            indices   = np.asarray(indices,   dtype=int).copy()
            distances = np.asarray(distances, dtype=float).copy()
            _knn_done = True
        except ImportError:
            pass

        if not _knn_done:
            nbrs = NearestNeighbors(n_neighbors=_n_query, algorithm='auto', n_jobs=-1).fit(X_pca)
            distances, indices = nbrs.kneighbors(X_pca)

    # Sort each row by (distance, index) for deterministic tie-breaking.
    _o1       = np.argsort(indices,   axis=1, kind='stable')
    distances = np.take_along_axis(distances, _o1, axis=1)
    indices   = np.take_along_axis(indices,   _o1, axis=1)
    _o2       = np.argsort(distances, axis=1, kind='stable')
    distances = np.take_along_axis(distances, _o2, axis=1)
    indices   = np.take_along_axis(indices,   _o2, axis=1)

    # Strip the query point (self) from every row, then keep n_neighbors entries.
    # Self appears at distance 0; after distance-sorting it is at column 0.
    # We check explicitly to handle any edge case where self might not be present.
    _rows      = np.arange(n_obs)
    _self_mask = (indices == _rows[:, None])   # (n_obs, n_query)

    clean_idx  = np.empty((n_obs, n_neighbors), dtype=int)
    clean_dist = np.empty((n_obs, n_neighbors), dtype=float)

    # Vectorised path: if self is always at column 0 (the common case), slice columns 1:
    if _self_mask[:, 0].all():
        clean_idx  = indices[:, 1:n_neighbors + 1]
        clean_dist = distances[:, 1:n_neighbors + 1]
    else:
        # Fallback: per-row removal for rows where self is not at column 0.
        clean_idx  = indices[:, 1:n_neighbors + 1].copy()
        clean_dist = distances[:, 1:n_neighbors + 1].copy()
        for i in np.where(~_self_mask[:, 0])[0]:
            sp_arr = np.where(_self_mask[i])[0]
            if sp_arr.size > 0:
                sp_pos = sp_arr[0]
                row_i  = np.delete(indices[i],   sp_pos)[:n_neighbors]
                row_d  = np.delete(distances[i], sp_pos)[:n_neighbors]
            else:
                row_i  = indices[i, :n_neighbors]
                row_d  = distances[i, :n_neighbors]
            clean_idx[i]  = row_i
            clean_dist[i] = row_d

    indices   = clean_idx
    distances = clean_dist
    # distances[:, 0] is now the distance to the true nearest non-self neighbor.

    # 1. Distance to nearest (true nearest non-self neighbor, matching R's distanceToNearest).
    dist_to_k = distances[:, 0].copy()
    # Replace any accidental zeros (identical cells) with the global minimum positive distance.
    if np.any(dist_to_k == 0):
        pos = dist_to_k[dist_to_k > 0]
        dist_to_k[dist_to_k == 0] = pos.min() if pos.size > 0 else 1e-6

    # 2. Ratio of artificial doublets in neighborhood at multiple k values.
    neighbor_types = y_type[indices]  # (n_obs, n_neighbors) — no self
    ratio_dict = {}
    k_vals = np.sort(np.unique([k for k in [3, 10, 15, 20, 25, 50, n_neighbors] if k <= n_neighbors]))
    for ki in k_vals:
        ratio_dict[f'ratio_doublets_{ki}'] = np.mean(neighbor_types[:, :ki], axis=1)

    # 3. Weighted density — mirrors R's dw = sqrt(k - seq_len(k)) * 1/dist.
    # With self removed, distances[:, 0] is the true nearest neighbor distance.
    # R replaces any zero in the first-neighbor column before computing weights.
    SAFE_DIST = distances.copy()
    _first_pos = SAFE_DIST[:, 0]
    _min_pos   = _first_pos[_first_pos > 0].min() if (_first_pos > 0).any() else 1e-6
    SAFE_DIST[SAFE_DIST == 0] = _min_pos

    # rank_weights: sqrt(k-1), sqrt(k-2), ..., 0  (matching R's sqrt(k - seq_len(k)))
    ranks        = np.arange(1, n_neighbors + 1)
    rank_weights = np.sqrt(n_neighbors - ranks)
    dist_weights = 1.0 / SAFE_DIST
    weights      = rank_weights * dist_weights
    row_sums     = weights.sum(axis=1, keepdims=True)
    norm_weights = weights / np.where(row_sums == 0, 1.0, row_sums)
    weighted_score = np.sum(neighbor_types * norm_weights, axis=1)

    # 4. Most likely origin & difficulty (unchanged logic, now using clean indices).
    unique_origins  = pd.unique(origins[~pd.isnull(origins)])
    origin_map      = {o: i for i, o in enumerate(unique_origins)}
    rev_origin_map  = {i: o for i, o in enumerate(unique_origins)}
    n_origins       = len(unique_origins)

    origin_ambiguous = np.full(n_obs, False, dtype=bool)
    final_origins    = np.full(n_obs, -1,    dtype=int)

    if n_origins > 0:
        s_origins        = pd.Series(origins)
        origins_num      = s_origins.map(origin_map).fillna(-1).astype(int).values
        neighbor_orig_num = origins_num[indices]  # (n_obs, n_neighbors)

        _cnt      = np.zeros((n_obs, n_origins), dtype=np.int32)
        for _j in range(n_origins):
            _cnt[:, _j] = (neighbor_orig_num == _j).sum(axis=1)

        _no_valid  = (neighbor_orig_num >= 0).sum(axis=1) == 0
        _mx_cnt    = _cnt.max(axis=1)
        _best_org  = np.argmax(_cnt, axis=1)
        _n_at_mx   = (_cnt == _mx_cnt[:, None]).sum(axis=1)

        _unique_max = (_n_at_mx == 1) & ~_no_valid
        final_origins[_unique_max]    = _best_org[_unique_max]
        origin_ambiguous[_unique_max] = False

        for i in np.where((_n_at_mx > 1) & ~_no_valid)[0]:
            _row       = neighbor_orig_num[i]
            _tied_vals = np.where(_cnt[i] == _mx_cnt[i])[0]
            _mins      = np.array([
                distances[i][_row == v].min() if (_row == v).any() else np.inf
                for v in _tied_vals
            ])
            _ord      = np.argsort(_mins)
            x1        = _mins[_ord[0]]
            x2        = _mins[_ord[1]] if len(_ord) >= 2 else np.inf
            ambiguous = (x1 <= 0) or not ((x2 - x1) / x1 > 0.2)
            final_origins[i]    = _tied_vals[_ord[0]]
            origin_ambiguous[i] = ambiguous

        most_likely_str = np.array(
            [rev_origin_map[i] if i >= 0 else np.nan for i in final_origins], dtype=object
        )
    else:
        most_likely_str = np.full(n_obs, np.nan, dtype=object)

    difficulty = np.ones(n_obs, dtype=float)
    if n_origins > 0:
        df_art       = pd.DataFrame({'origin': origins, 'weighted': weighted_score, 'type': y_type})
        df_art       = df_art[df_art['type'] == 1]
        origin_means = df_art.groupby('origin')['weighted'].mean()
        mapped_means = origin_means.reindex(most_likely_str).values
        valid_means  = ~np.isnan(mapped_means)
        difficulty[valid_means] = 1.0 - mapped_means[valid_means]

    # 5. Distance to nearest real / doublet neighbor (self already absent).
    # Fallback for cells with no real/doublet neighbour in k: use 2 × max nearest-neighbour
    # distance, matching R's `md <- max(knn$distance[,1]); dB <- 2*md`.
    max_dist                = dist_to_k.max() * 2
    dist_to_nearest_real    = np.where(neighbor_types == 0, distances, max_dist).min(axis=1)
    dist_to_nearest_doublet = np.where(neighbor_types == 1, distances, max_dist).min(axis=1)

    # 6. nAbove2: count of genes with counts > 2 (raw counts, not KNN-derived).
    counts_matrix = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    if sp.issparse(counts_matrix):
        n_above2 = np.asarray((counts_matrix > 2).sum(axis=1)).ravel()
    else:
        n_above2 = np.sum(counts_matrix > 2, axis=1)

    res = {
        'distance_to_nearest': dist_to_k,
        'weighted_density': weighted_score,
        'distance_to_nearest_real': dist_to_nearest_real,
        'distance_to_nearest_doublet': dist_to_nearest_doublet,
        'nAbove2': n_above2,
        'difficulty': difficulty,
        'most_likely_origin': most_likely_str,
        'origin_ambiguous': origin_ambiguous,
    }
    res.update(ratio_dict)
    try:
        adata.obsm['knn_indices']   = indices
        adata.obsm['knn_distances'] = distances
    except Exception:
        pass
    return res
