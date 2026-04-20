import numpy as np
import scipy.sparse as sp
from scipy.stats import binom
import math
import pandas as pd
import warnings
import scanpy as sc

def cxds2(adata, which_dbls=None, n_top=500, bin_thresh=None, verbose=False):
    """
    Calculates a coexpression-based doublet score (CXDS).
    
    Implementation of the cxds2 method from scDblFinder (originally from scds).
    Scores cells based on the co-expression of gene pairs that are typically 
    mutually exclusive in the singlet population.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Expects counts in `adata.X` or `adata.layers['counts']`.
    which_dbls : list or np.ndarray, optional
        Indices of cells that are known doublets (e.g. simulated).
        These cells are excluded from the gene pair learning step but are scored.
    n_top : int, optional
        Number of top variable genes to use for scoring. Default is 500.
    bin_thresh : float, optional
         Threshold for binarizing counts. If None, estimated from data.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    np.ndarray
        Array of CXDS scores for each cell.
    """
    
    # 1. Get counts
    if 'counts' in adata.layers:
        X = adata.layers['counts']
    else:
        # Assuming X contains counts if no layer specified, or raw data
        X = adata.X
    
    # Ensure sparse csc or csr for efficiency, but we need slicing
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    
    X = X.copy()
    # Handle NA? Sparse matrix usually doesn't have NA. R code: x[is.na(x)] <- 0L
    # In python sparse, assume data is valid numbers.
    
    n_cells = X.shape[0]
    n_genes = X.shape[1]

    # 2. Binarization Threshold
    # R: if(is.null(binThresh)) ...
    if bin_thresh is None:
        # Check density
        n_nonzero = X.nnz
        p_nonzero = n_nonzero / (n_cells * n_genes)
        
        if p_nonzero > 0.5:
            # Dense data strategy
            # R: pNonZero <- rowSums(x>0)/ncol(x) -> proportion of cells expressing each gene?
            # Wait, `rowSums(x>0)/ncol(x)` in R where x is (genes x cells)
            # Python X is (cells x genes). So `colSums(X>0)/n_cells`.
            
            # X > 0
            if sp.issparse(X):
                # Efficient calculation of col sums of binary
                # X.getnnz(axis=0) returns Number of stored values, including explicit zeros.
                # Need explicit check for > 0
                X_bin_temp = (X > 0).astype(int)
                gene_p_nonzero = np.array(X_bin_temp.sum(axis=0)).flatten() / n_cells
            else:
                 gene_p_nonzero = np.sum(X > 0, axis=0) / n_cells
            
            # Keep top genes with lowest sparsity (highest zeros? no head(order(pNonZero)))
            # R: head(order(pNonZero), ntop) 
            # order(p) gives index of smallest p first.
            # So genes with LOWEST proportion of non-zeros (most sparse).
            # Wait, if pNonZero > 0.5 (dense), we select genes that are sparse?
            # Yes, `head(order(pNonZero))` selects smallest values.
            
            # We are not subsetting genes yet here in R code, just x calculation for quantile?
            # R: x <- x[head(order(pNonZero), ntop),] 
            # Then median of that subset.
            
            # Let's replicate selecting subset for threshold estimation
            gene_indices = np.argsort(gene_p_nonzero)[:n_top]
            X_subset = X[:, gene_indices]
            
            # binThresh <- max(1L, as.numeric(quantile(x@x, mean(pNonZero)*0.5)))
            # quantile of ALL data points in subset? 
            # R: quantile(x@x, ...) checks non-zero values.
            if sp.issparse(X_subset):
                 data_values = X_subset.data
            else:
                 data_values = X_subset[X_subset > 0]
                 
            mean_p = np.mean(gene_p_nonzero) # R: mean(pNonZero) (of all genes or subset? R uses pNonZero which is vector of all)
            
            if len(data_values) > 0:
                 threshold_val = np.quantile(data_values, mean_p * 0.5)
                 bin_thresh = max(1, int(threshold_val))
            else:
                 bin_thresh = 1
        else:
            bin_thresh = 1
            
    if verbose:
        print(f"Binarization threshold: {bin_thresh}")

    # 3. Binarize
    # R: Bp <- x <- x >= binThresh
    # Python X is (cells x genes).
    if sp.issparse(X):
        # Efficient binarization
        # Create boolean mask
        X_bin = (X >= bin_thresh).astype(float) # Keep as float/int for matrix mult? 
        # R uses logical for Bp. Logic operations used later.
        # But later `Bp %*% ...` implies numeric (0/1).
        X_bin = X_bin.tocsc() # Convert to CSC for column operations if needed, or CSR.
    else:
        X_bin = (X >= bin_thresh).astype(float)

    # 4. Filter Genes (HVG)
    # R: ps <- rowMeans(x) (genes x cells -> mean across cells)
    # Python: mean across cells (axis 0)
    
    if sp.issparse(X_bin):
        ps = np.array(X_bin.mean(axis=0)).flatten()
    else:
        ps = X_bin.mean(axis=0)
        
    # R: hvg <- order(ps * (1 - ps), decreasing=TRUE)[seq_len(ntop)]
    variances = ps * (1 - ps)
    # argsort is ascending. use [::-1]
    hvg_indices = np.argsort(variances)[::-1][:n_top]
    
    # Subset
    # R: Bp <- x <- x[hvg, ]
    # Python X is (cells x genes), so column subset
    X_bin = X_bin[:, hvg_indices]
    ps = ps[hvg_indices]
    
    # 5. Handle whichDbls for Learning
    # R: if(length(whichDbls)>0) Bp <- Bp[,-whichDbls] (cols are cells)
    # Python: Remove ROWS corresponding to which_dbls
    
    if which_dbls is not None and len(which_dbls) > 0:
        # Create mask for training set
        is_train = np.ones(n_cells, dtype=bool)
        is_train[which_dbls] = False
        Bp_train = X_bin[is_train, :]
    else:
        Bp_train = X_bin
        
    num_train_cells = Bp_train.shape[0] # R: size=ncol(Bp) which is # cells
    
    # Recompute ps on Bp_train only!
    if sp.issparse(Bp_train):
        ps = np.array(Bp_train.mean(axis=0)).flatten()
    else:
        ps = Bp_train.mean(axis=0)
    
    # 6. Expected probabilities
    # R: prb <- outer(ps, 1 - ps)
    # prb <- prb + t(prb)
    # ps is vector of length ntop
    
    prb = np.outer(ps, 1 - ps)
    prb = prb + prb.T
    
    # 7. Observed counts (Discordant pairs)
    # R: obs <- Bp %*% (1 - Matrix::t(Bp))
    # R Bp is (genes x cells).
    # Python Bp_train is (cells x genes).
    # So equivalent is Bp_train.T (genes x cells).
    # R calc: Bp @ (1 - Bp.T) 
    # = Bp @ 1 - Bp @ Bp.T
    # Dimensions: (G x C) @ (C x G) = G x G
    
    # Let's compute using Python logic derived earlier:
    # O_ij = n_i + n_j - 2 K_ij
    # where K = Bp_train.T @ Bp_train (G x G co-occurrence)
    # n_i = sum of Bp_train.T (each row sum = gene total count)
    
    # K matrix (intersection count)
    # Bp_train is (Cells x Genes).
    # K = Bp_train.T @ Bp_train -> (G x G)
    # Note: Bp_train is sparse, so dot product is efficient.
    
    if sp.issparse(Bp_train):
        K = (Bp_train.T @ Bp_train).toarray() # G is small (500), so dense GxG is fine
        n_counts = np.array(Bp_train.sum(axis=0)).flatten()
    else:
        K = Bp_train.T @ Bp_train
        n_counts = Bp_train.sum(axis=0)
        
    # obs[i,j] = n_i + n_j - 2*K_ij
    # Use broadcasting
    # n_i (G,) + n_j (G,) -> GxG
    N_matrix = np.add.outer(n_counts, n_counts)
    obs = N_matrix - 2 * K
    
    # R: obs <- obs + Matrix::t(obs)
    # My derivation `obs = (n_i - K) + (n_j - K)` includes both terms?
    # R's `obs` intermediate:
    # A = Bp @ (1 - Bp.T)
    # A_ij = sum_k Bp[i,k] * (1 - Bp[j,k]) = n_i - K_ij
    # Then `obs <- A + A.T`.
    # A.T_ij = A_ji = n_j - K_ji = n_j - K_ij
    # So obs_final_ij = (n_i - K_ij) + (n_j - K_ij) = n_i + n_j - 2*K_ij.
    # Matches!
    
    # 8. Binomial Probability (Score S)
    # R: S <- pbinom(obs - 1, prob = prb, size=ncol(Bp), lower.tail=FALSE, log.p=TRUE)
    # size = num_train_cells
    
    # Check for zero S (impossible return)
    # R: if(all(S==0)) return 0
    
    # Compute S using log survival function
    # log(P(X > obs - 1)) = log(P(X >= obs)) = logsf(obs - 1)
    # Note: obs is integer counts.
    
    # Make sure inputs are arrays
    # obs can be float if K came from float mult? Round to be safe
    obs = np.round(obs)
    
    # Avoid log(0) issues? logsf handles it (-inf).
    S = binom.logsf(obs - 1, n=num_train_cells, p=prb)
    
    if np.all(S == 0):
        return np.zeros(n_cells)
    
    # R: if(any(w <- is.infinite(S))){ smin <- min(S[!is.infinite(S)]); S[S<smin] <- smin }
    # S contains log-probs, so expected range (-inf, 0].
    # -inf happens if event is extremely unlikely (p-value ~ 0).
    # R clamps -inf to minimum finite value.
    
    is_inf = np.isinf(S)
    if np.any(is_inf):
        finite_vals = S[~is_inf]
        if len(finite_vals) > 0:
            smin = np.min(finite_vals)
            S[is_inf] = smin
        else:
            # All infinite? obscure case
            S[:] = -1e10 # Arbitrary small
            
    # 9. Calculate Cell Scores
    # R: s <- -Matrix::colSums(x * (S %*% x))
    # x here is the original binary matrix (including all cells), subset to hvg genes.
    # Python X_bin (Cells x Genes).
    # R's x is (Genes x Cells).
    
    # Formula inner term: S (GxG) %*% x (GxC) -> (GxC)
    # Then x * (S %*% x) -> element-wise product (GxC)
    # colSums -> vector of size C.
    
    # Python equivalent:
    # Term = X_bin @ S (Cells x Genes @ Genes x Genes) -> (Cells x Genes)
    # Note S is symmetric. (S %*% x) in R is equivalent to (x.T @ S).T = S.T @ x = S @ x.
    # R: x is gene x cell.
    # Python: term = X_bin @ S. (C x G).
    # Then element-wise multiply by X_bin: X_bin * (X_bin @ S)
    # Then sum across genes (axis 1).
    
    # Logic check:
    # R: colSums(x * (S %*% x))
    # Cell k: sum_i [ x_ik * sum_j (S_ij * x_jk) ]
    # = sum_i sum_j S_ij x_ik x_jk
    
    # Python:
    # (X_bin @ S) -> M, M_kj = sum_l X_kl S_lj
    # (X_bin * M) -> elementwise. P_kj = X_kj * M_kj = X_kj * sum_l X_kl S_lj
    # Sum axis 1 (genes): sum_j P_kj = sum_j X_kj * sum_l X_kl S_lj
    # = sum_j sum_l S_lj X_kj X_kl
    # Matches! (indices swapped but symmetric S makes it same).
    
    # Compute:
    # X_bin is sparse. S is dense.
    # X_bin @ S -> Dense (C x G) usually (unless S is sparse, which it isn't here).
    
    Proj = X_bin @ S # (C x G) from (C x G) @ (G x G)
    
    # Element-wise multiply X_bin and Proj.
    # Since X_bin is binary (0/1), we just need to sum Proj values where X_bin is 1.
    # If X_bin is CSR/CSC, we can just mask Proj? Or multiply.
    
    if sp.issparse(X_bin):
        # Sparse * Dense -> Sparse? Or Dense?
        # Scipy sparse multiply is elementwise.
        # X_bin.multiply(Proj) returns sparse matrix.
        product = X_bin.multiply(Proj)
        scores = np.array(product.sum(axis=1)).flatten()
    else:
        scores = np.sum(X_bin * Proj, axis=1)
        
    s = -scores
    
    # Normalization
    # R: s <- s - min(s)
    # R: s/max(s)
    
    s = s - s.min()
    if s.max() > 0:
        s = s / s.max()
        
    return s

def select_features(adata, clusters=None, n_features=1000, prop_markers=0.0, fdr_max=0.05):
    """
    Selects top features/genes based on overall expression and cluster markers.
    Mimics selFeatures.R from scDblFinder.
    """
    n_vars = adata.n_vars
    var_names = np.array(adata.var_names)
    
    if n_vars <= n_features:
        return list(var_names)
        
    if clusters is None:
        prop_markers = 0.0
        
    g_idx = []
    ng = math.ceil((1 - prop_markers) * n_features)
    
    X = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    
    if ng > 0:
        if clusters is None:
            # global rowMeans (which is column means for Python AnnData)
            means = np.asarray(X.mean(axis=0)).flatten()
            g_idx = np.argsort(means)[::-1][:ng].tolist()
        else:
            try:
                # Calculate cluster means
                unique_clusters = np.unique(clusters)
                n_clusters = len(unique_clusters)
                
                cl_means = np.zeros((n_vars, n_clusters))
                for i, cl in enumerate(unique_clusters):
                    mask = (clusters == cl)
                    cl_means[:, i] = np.asarray(X[mask].mean(axis=0)).flatten()
                
                # Equivalent to R's apply(cl.means, 2, order)[seq_len(n_features)]
                cl_orders = np.zeros((n_features, n_clusters), dtype=int)
                for i in range(n_clusters):
                    cl_orders[:, i] = np.argsort(cl_means[:, i])[::-1][:n_features]
                
                # t().as.numeric() flattens row by row (interleaving clusters)
                cl_orders_flat = cl_orders.flatten()
                # unique preserving order
                _, idx = np.unique(cl_orders_flat, return_index=True)
                g_idx = cl_orders_flat[np.sort(idx)][:ng].tolist()
                
            except Exception as e:
                means = np.asarray(X.mean(axis=0)).flatten()
                g_idx = np.argsort(means)[::-1][:ng].tolist()
                
    if ng == n_features:
        return [var_names[i] for i in g_idx]
        
    # Marker genes part using scanpy rank_genes_groups
    adata_tmp = sc.AnnData(X, obs={'clusters': pd.Categorical(clusters)})
    adata_tmp.var_names = var_names
    
    try:
        sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)
        sc.tl.rank_genes_groups(adata_tmp, groupby='clusters', method='wilcoxon', key_added='markers')
        
        marker_results = []
        for cl in pd.unique(adata_tmp.obs['clusters']):
            r = sc.get.rank_genes_groups_df(adata_tmp, group=cl, key='markers')
            r = r[r['pvals_adj'] < fdr_max].copy()
            # mimic 'Top' ranking
            r['Top'] = np.arange(1, len(r) + 1)
            marker_results.append(r)
            
        g_names = [var_names[i] for i in g_idx]
            
        if marker_results:
            mm = pd.concat(marker_results)
            # R equivalent: g2 <- unique(c(g, mm$gene))
            mm_genes = mm['names'].tolist()
            
            seen = set(g_names)
            g2 = list(g_names)
            for name in mm_genes:
                if name not in seen:
                    g2.append(name)
                    seen.add(name)
                    
            if len(g2) < n_features:
                return g2
                
            i = n_features / (2 * len(pd.unique(adata_tmp.obs['clusters'])))
            
            while True:
                mm_filtered_genes = mm[mm['Top'] <= i]['names'].tolist()
                
                seen_g = set(g_names)
                current_g = list(g_names)
                for name in mm_filtered_genes:
                    if name not in seen_g:
                        current_g.append(name)
                        seen_g.add(name)
                        
                if len(current_g) >= n_features:
                    return current_g[:n_features]
                i += 1
                
    except Exception as e:
        print(f"Warning: Marker selection failed, returning fallback top expressed genes. {e}")
        means = np.asarray(X.mean(axis=0)).flatten()
        g_idx_fallback = np.argsort(means)[::-1][:n_features].tolist()
        return [var_names[i] for i in g_idx_fallback]
        
    return g_names[:n_features]
