import numpy as np
import scipy.sparse as sp
from scipy.stats import binom
import math
import pandas as pd
import warnings
import scanpy as sc
from statsmodels.stats.multitest import fdrcorrection

def cxds2(adata, which_dbls=None, ntop=500, bin_thresh=None, verbose=False, n_top=None):
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
    
    # R uses genes x cells; transpose AnnData-style cells x genes into that layout.
    if hasattr(adata, 'layers'):
        if 'counts' in adata.layers:
            X = adata.layers['counts']
        else:
            X = adata.X
    else:
        X = adata

    if not sp.issparse(X):
        X = np.asarray(X)
        X = sp.csr_matrix(X)

    # Internal representation follows R exactly: genes x cells.
    x = X.T.tocsr().copy()
    x.data = np.asarray(x.data)
    x[x != x] = 0

    if n_top is not None:
        ntop = n_top

    n_genes, n_cells = x.shape

    if bin_thresh is None:
        if sp.issparse(x):
            # For sparse matrices, .size is NOT total elements; calculate correctly
            total_size = x.shape[0] * x.shape[1]
            p_nonzero = x.nnz / total_size
        else:
            p_nonzero = np.sum(x > 0) / x.size

        if p_nonzero > 0.5:
            if sp.issparse(x):
                p_nonzero_vec = np.asarray((x > 0).sum(axis=1)).ravel() / n_cells
            else:
                p_nonzero_vec = np.sum(x > 0, axis=1) / n_cells

            # Match R's stable `order()` tie-breaking: preserve original gene order on ties.
            order = np.lexsort((np.arange(p_nonzero_vec.shape[0]), p_nonzero_vec))
            keep = order[:ntop]
            x = x[keep, :]
            p_nonzero_vec = p_nonzero_vec[keep]

            if sp.issparse(x):
                vals = x.data
                if vals.size > 0:
                    bin_thresh = max(1, float(np.quantile(vals, np.mean(p_nonzero_vec) * 0.5)))
                else:
                    bin_thresh = 1
            else:
                bin_thresh = max(1, float(np.median(np.asarray(x))))
        else:
            bin_thresh = 1

    if verbose:
        print(f"Binarization threshold: {bin_thresh}")

    x = (x >= bin_thresh).astype(float)
    ps = np.asarray(x.mean(axis=1)).ravel() if sp.issparse(x) else np.mean(x, axis=1)

    if x.shape[0] > ntop:
        score = ps * (1 - ps)
        # Match R's order(): sort descending by score, with later indices first on ties.
        # Due to subtle differences in tie-breaking between numpy and R's order(),
        # we use the most closely-matching sort: (score, indices) reversed
        # This gives near-perfect parity with R in most cases
        sorted_idx = np.lexsort((np.arange(score.shape[0]), score))[::-1]
        hvg = sorted_idx[:ntop]
        x = x[hvg, :]
        ps = ps[hvg]

    Bp = x
    if which_dbls is not None and len(which_dbls) > 0:
        keep_cols = np.ones(Bp.shape[1], dtype=bool)
        keep_cols[np.asarray(which_dbls, dtype=int)] = False
        Bp = Bp[:, keep_cols]

    prb = np.outer(ps, 1 - ps)
    prb = prb + prb.T

    if sp.issparse(Bp):
        K = (Bp @ Bp.T).toarray()
        n_counts = np.asarray(Bp.sum(axis=1)).ravel()
    else:
        K = Bp @ Bp.T
        n_counts = np.sum(Bp, axis=1)
    obs = np.add.outer(n_counts, n_counts) - 2 * K

    S = binom.logsf(np.round(obs) - 1, n=Bp.shape[1], p=prb)

    if np.all(S == 0):
        return np.zeros(n_cells)

    finite = np.isfinite(S)
    if np.any(~finite):
        smin = np.min(S[finite])
        S[S < smin] = smin

    if sp.issparse(x):
        product = x.multiply(S @ x)
        scores = np.asarray(product.sum(axis=0)).ravel()
    else:
        scores = np.sum(x * (S @ x), axis=0)

    s = -scores
    s = s - np.min(s)
    if np.max(s) > 0:
        s = s / np.max(s)

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
        
    # Marker genes part using a binomial-style test on raw counts.
    # This mirrors scran::findMarkers(..., test.type="binom", assay.type="counts").
    try:
        clusters_arr = np.asarray(clusters)
        unique_clusters = pd.unique(clusters_arr)
        g_names = [var_names[i] for i in g_idx]

        marker_results = []
        if sp.issparse(X):
            X_expr = X > 0
        else:
            X_expr = np.asarray(X > 0)

        total_expr = np.asarray(X_expr.sum(axis=0)).ravel() if sp.issparse(X_expr) else np.sum(X_expr, axis=0)
        total_cells = X.shape[0]
        bg_prob = np.clip(total_expr / max(total_cells, 1), 1e-12, 1 - 1e-12)

        for cl in unique_clusters:
            mask = clusters_arr == cl
            n_cl = int(np.sum(mask))
            if n_cl == 0:
                continue

            if sp.issparse(X_expr):
                expr_in_cl = np.asarray(X_expr[mask].sum(axis=0)).ravel()
            else:
                expr_in_cl = np.sum(X_expr[mask], axis=0)

            pvals = binom.sf(expr_in_cl - 1, n_cl, bg_prob)
            pvals = np.asarray(pvals, dtype=float)
            pvals[~np.isfinite(pvals)] = 1.0
            _, fdr_vals = fdrcorrection(pvals, alpha=fdr_max)

            order = np.argsort(pvals)
            r = pd.DataFrame({
                'names': var_names[order],
                'Top': np.arange(1, len(order) + 1),
                'FDR': fdr_vals[order],
            })
            r = r[r['FDR'] < fdr_max].copy()
            marker_results.append(r)

        if marker_results:
            mm = pd.concat(marker_results, ignore_index=True)
            mm_genes = mm['names'].tolist()

            seen = set(g_names)
            g2 = list(g_names)
            for name in mm_genes:
                if name not in seen:
                    g2.append(name)
                    seen.add(name)

            if len(g2) < n_features:
                return g2

            i = n_features / (2 * len(unique_clusters))
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
    