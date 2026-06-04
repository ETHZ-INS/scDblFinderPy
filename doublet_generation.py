import numpy as np
import scipy.sparse as sp
import warnings
import pandas as pd
from anndata import AnnData
import scanpy as sc
import itertools
import random
from .r_sample_emulation import sample_int_r
from .rng import coerce_rng, CentralRNG


def _coerce_rng(random_state=None, rng=None):
    # Return a CentralRNG instance coerced from inputs.
    return coerce_rng(random_state=random_state, rng=rng)


def _python_random_from_rng(rng):
    # Accept CentralRNG, numpy Generator, or RandomState
    if hasattr(rng, "integers"):
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    elif hasattr(rng, 'randint32'):
        seed = int(rng.randint32())
    else:
        seed = int(getattr(rng, 'randint', lambda a, b: 0)(0, 2**32 - 1))
    return random.Random(seed)


def _sample_python(values, size, rng, replace=False):
    values = np.asarray(values)
    if size <= 0 or values.size == 0:
        return np.array([], dtype=values.dtype)
    # Use CentralRNG or derive a RandomState for Python's sampling if needed
    if isinstance(rng, CentralRNG) or hasattr(rng, 'randint32'):
        sample_rng = rng
    elif hasattr(rng, 'integers'):
        # derive deterministic RandomState from provided Generator
        sample_rng = np.random.RandomState(int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)))
    else:
        sample_rng = np.random.RandomState(int(rng))

    idx = sample_int_r(len(values), size, replace=replace, rng=sample_rng) - 1
    if replace:
        return values[idx]
    return values[idx]

def create_doublets(X, dbl_idx, clusters=None, resamp=0.5, half_size=0.5, adjust_size=False, random_state=0, rng=None, debug_outdir=None):
    """
    Creates artificial doublet cells by combining given pairs of cells.
    
    Port of R function `createDoublets`.
    
    Parameters
    ----------
    X : scipy.sparse.spmatrix
        Count matrix of real cells (Cells x Genes). Should be in CSR/CSC format.
    dbl_idx : np.ndarray
        Numpy array of shape (N_doublets, 2) containing indices of cell pairs to combine.
    clusters : np.ndarray or list, optional
        Vector of cluster labels for each row of X. Required if adjust_size is True.
    resamp : float or bool
        Whether to resample the doublets using the poisson distribution. 
        If a number between 0 and 1, the proportion of doublets to resample.
    half_size : float or bool
        Whether to half the library size of doublets. 
        If a number between 0 and 1, the proportion of doublets for which to perform the adjustment.
    adjust_size : float or bool
        Whether to adjust the size of the doublets using the median sizes per cluster.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix of artificial doublets (N_doublets x N_genes).
    """

    rng = _coerce_rng(random_state=random_state, rng=rng)
    sample_rng = rng
    
    def check_prop_arg(arg):
        if isinstance(arg, bool):
            return 1.0 if arg else 0.0
        return float(arg)

    adjust_size_val = check_prop_arg(adjust_size)
    half_size_val = check_prop_arg(half_size)
    resamp_val = check_prop_arg(resamp)

    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    if not 0 <= adjust_size_val <= 1:
        raise ValueError("adjust_size should be a bool or number between 0 and 1")
    if not 0 <= half_size_val <= 1:
        raise ValueError("half_size should be a bool or number between 0 and 1")

    n_doublets = dbl_idx.shape[0]
    
    # 1. Determine which doublets get size adjustment
    # In R: sample.int(nrow(dbl.idx), size=round(adjustSize*nrow(dbl.idx)))
    n_adjust = int(round(adjust_size_val * n_doublets))
    
    if n_adjust > 0:
        w_ad = sample_int_r(n_doublets, n_adjust, replace=False, rng=sample_rng) - 1
    else:
        w_ad = np.array([], dtype=int)
         
    idx1 = dbl_idx[:, 0]
    idx2 = dbl_idx[:, 1]

    # Debug: dump incoming pair indices
    if debug_outdir is not None:
        try:
            import os
            os.makedirs(debug_outdir, exist_ok=True)
            import pandas as pd
            df_in = pd.DataFrame({'idx1': idx1, 'idx2': idx2})
            df_in.to_csv(os.path.join(debug_outdir, 'create_doublets_incoming_pairs.csv'), index=False)
        except Exception:
            pass
    
    # Basic summation (Cell A + Cell B)
    X_dbl = X[idx1] + X[idx2]

    # 2. Check logic for Size Adjustment (adjustSize)
    if n_adjust > 0:
        if clusters is None:
            raise ValueError("If `adjust_size` is True/ >0, clusters must be given.")
        
        clusters = np.array(clusters)
        ls = np.array(X.sum(axis=1)).flatten()
        unique_clusters = np.unique(clusters)
        
        # Calculate median library size per cluster
        csz = {}
        for c in unique_clusters:
            mask = (clusters == c)
            if np.any(mask):
                csz[c] = np.median(ls[mask])
            else:
                csz[c] = 0
            
        # Helper to do size adjustment math
        # We process ONLY the rows that need adjustment (w_ad)
        idx_to_adj_in_dbl = w_ad
        
        adj_idx1 = idx1[idx_to_adj_in_dbl]
        adj_idx2 = idx2[idx_to_adj_in_dbl]
        
        c1 = clusters[adj_idx1]
        c2 = clusters[adj_idx2]
        ls1 = ls[adj_idx1]
        ls2 = ls[adj_idx2]

        # Calculate weighting factor
        # Avoid division by zero
        denom_cells = ls1 + ls2
        with np.errstate(divide='ignore', invalid='ignore'):
            ls_ratio = ls1 / denom_cells
            ls_ratio[denom_cells == 0] = 0.5

        ls1_median = np.array([csz[c] for c in c1])
        ls2_median = np.array([csz[c] for c in c2])
        
        denom_clusters = ls1_median + ls2_median
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_median = ls1_median / denom_clusters
            ratio_median[denom_clusters == 0] = 0.5 
            
        factor = (ls_ratio + ratio_median) / 2
        factor = np.clip(factor, 0.2, 0.8)
        
        target_ls = ls1 + ls2
        
        # Extract source rows again for weighting
        X1_sub = X[adj_idx1]
        X2_sub = X[adj_idx2]
        
        # weighted combination
        X1_weighted = X1_sub.multiply(factor[:, None])
        X2_weighted = X2_sub.multiply((1 - factor)[:, None])
        
        X2_combined = X1_weighted + X2_weighted
        
        # Re-normalize to target library size
        current_sums = np.array(X2_combined.sum(axis=1)).flatten()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factors = target_ls / current_sums
            scale_factors[current_sums == 0] = 1.0 
        
        X2_final = X2_combined.multiply(scale_factors[:, None])
        
        # Place back into the main doublet matrix
        X_dbl = X_dbl.tolil()
        X_dbl[idx_to_adj_in_dbl, :] = X2_final
        X_dbl = X_dbl.tocsr()

    # 3. Handle Half Size (halfSize)
    w_half = np.array([], dtype=int)
    if half_size_val > 0:
        n_half = int(np.ceil(half_size_val * n_doublets))
        if n_half > 0:
            w_half = sample_int_r(n_doublets, n_half, replace=False, rng=sample_rng) - 1
            coeffs = np.ones(n_doublets)
            coeffs[w_half] = 0.5
            scaler = sp.diags(coeffs)
            X_dbl = scaler @ X_dbl

    # 4. Handle Resampling (resamp)
    if resamp_val > 0:
        # R Logic: if(resamp!=halfSize) wAd <- sample.int(...)
        # This implies: if resamp == halfSize, we REUSE w_half from above.
        
        if abs(resamp_val - half_size_val) < 1e-9 and len(w_half) > 0:
             w_resamp = w_half
        else:
             n_resamp = int(np.ceil(resamp_val * n_doublets))
             if n_resamp > 0:
                 w_resamp = sample_int_r(n_doublets, n_resamp, replace=False, rng=sample_rng) - 1
             else:
                 w_resamp = np.array([], dtype=int)
        
        if len(w_resamp) > 0:
            sub_X = X_dbl[w_resamp].toarray()
            sub_X = rng.poisson(sub_X)
            
            X_dbl = X_dbl.tolil()
            X_dbl[w_resamp, :] = sp.csr_matrix(sub_X)
            X_dbl = X_dbl.tocsr()
    else:
        # If no resampling, round to nearest integer
        if sp.issparse(X_dbl):
             X_dbl.data = np.round(X_dbl.data)
        else:
             X_dbl = X_dbl.round()
    
    # Debug: dump resulting doublet indices shape
    if debug_outdir is not None:
        try:
            import os
            os.makedirs(debug_outdir, exist_ok=True)
            with open(os.path.join(debug_outdir, 'create_doublets_result_shape.txt'), 'w') as f:
                f.write(f"n_doublets={X_dbl.shape[0]}, n_genes={X_dbl.shape[1]}\n")
        except Exception:
            pass
    # Ensure integer counts and CSR format to match R's integer matrix behavior
    if sp.issparse(X_dbl):
        # Round any non-integer entries and cast to integer type
        try:
            X_dbl.data = np.rint(X_dbl.data).astype(np.int64)
        except Exception:
            X_dbl = sp.csr_matrix(np.rint(X_dbl.toarray()).astype(np.int64))
        X_dbl = X_dbl.tocsr()
    else:
        X_dbl = sp.csr_matrix(np.rint(np.asarray(X_dbl)).astype(np.int64))

    return X_dbl

def get_expected_doublets(clusters, dbr=None, only_heterotypic=True, dbr_per_1k=0.008):
    """
    Calculates the expected number of doublets for each cluster combination.
    """
    clusters = np.asarray(clusters)
    n_cells = len(clusters)
    if dbr is None:
        dbr = (dbr_per_1k * n_cells / 1000)
    
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    if len(unique_clusters) == 1:
        return {str(unique_clusters[0]): n_cells * dbr}

    proportions = counts / n_cells
    
    # Outer product: prop_i * prop_j
    prob_mat = np.outer(proportions, proportions)
    expected_mat = prob_mat * dbr * n_cells
    
    res = {}
    
    # Map back indices to cluster names
    idx_to_cluster = {i: c for i, c in enumerate(unique_clusters)}

    # Loop to create keys like "ClusterA+ClusterB"
    for i in range(len(unique_clusters)):
        for j in range(len(unique_clusters)):
            c1 = idx_to_cluster[i]
            c2 = idx_to_cluster[j]
            
            if only_heterotypic:
                if i >= j: continue
                # Heterotypic: double the probability (A+B and B+A are same doublet type)
                val = 2 * expected_mat[i, j]
                res[f"{c1}+{c2}"] = val
            else:
                if i > j: continue
                val = expected_mat[i, j]
                if i != j: val *= 2
                res[f"{c1}+{c2}"] = val
               
    return res

def get_cell_pairs(clusters, n=1000, ls=None, q=(0.1, 0.9), sel_mode="proportional", soft_min=5, random_state=0, rng=None, debug_outdir=None, use_sample_int=True):
    """
    Given a vector of cluster labels, returns pairs of cross-cluster cell indices.
    """
    rng = _coerce_rng(random_state=random_state, rng=rng)
    sample_rng = rng
    clusters = np.asarray(clusters)
    n_cells = len(clusters)
    indices = np.arange(n_cells)
    
    unique_clusters = np.unique(clusters)
    cli = {c: indices[clusters == c] for c in unique_clusters}
        
    # Filter by library size if provided
    if ls is not None:
        ls = np.asarray(ls)
        for c in cli:
            ls_c = ls[cli[c]]
            if len(ls_c) > 0:
                bounds = np.quantile(ls_c, q)
                mask = (ls_c >= bounds[0]) & (ls_c <= bounds[1])
                cli[c] = cli[c][mask]

    # Generate all pairs of clusters
    cluster_pairs = []
    uc_list = list(unique_clusters)
    
    for i, c1 in enumerate(uc_list):
        for j, c2 in enumerate(uc_list):
            if i < j: # Strictly heterotypic pairs for now
                cluster_pairs.append((c1, c2))
    
    if not cluster_pairs:
        return np.empty((0, 2), dtype=int), []

    # Calculate number of pairs
    pair_counts = {}
    
    if sel_mode == "uniform":
        count_per_pair = int(np.ceil(n / len(cluster_pairs)))
        for pair in cluster_pairs:
            pair_counts[pair] = count_per_pair
    else:
        # Follow R's exact scaling logic:
        # 1) ed <- getExpectedDoublets(...)
        # 2) ed <- ed * n / sum(ed)
        # 3) if selMode=="sqrt": ed <- sqrt(ed)
        # 4) ed <- soft_min + ed
        # 5) final <- ceiling(ed * n / sum(ed))
        ed_dict = get_expected_doublets(clusters, only_heterotypic=True)
        ed_keys = [f"{p[0]}+{p[1]}" for p in cluster_pairs]
        ed_vals = np.array([ed_dict.get(k, 0.0) for k in ed_keys], dtype=float)
        total_ed = ed_vals.sum()
        if total_ed == 0:
            ed_scaled = np.full_like(ed_vals, 1.0)
        else:
            ed_scaled = ed_vals * (n / total_ed)

        if sel_mode == "sqrt":
            ed_scaled = np.sqrt(ed_scaled)

        ed_scaled = soft_min + ed_scaled

        # final normalization and ceiling as in R
        denom = ed_scaled.sum()
        if denom == 0:
            final_counts = np.ceil(ed_scaled)
        else:
            final_counts = np.ceil(ed_scaled * (n / denom)).astype(int)

        for pair, cnt in zip(cluster_pairs, final_counts):
            pair_counts[pair] = int(cnt)

    all_pairs = []
    all_origins = []

    for pair in cluster_pairs:
        count = pair_counts[pair]
        if count <= 0: continue
        
        c1, c2 = pair
        cells1 = cli[c1]
        cells2 = cli[c2]
        
        if len(cells1) == 0 or len(cells2) == 0:
            continue
            
        # Sample with replacement if needed
        if use_sample_int:
            pos1 = sample_int_r(len(cells1), count, replace=(count > len(cells1)), rng=sample_rng) - 1
            pos2 = sample_int_r(len(cells2), count, replace=(count > len(cells2)), rng=sample_rng) - 1
            idx1 = cells1[pos1]
            idx2 = cells2[pos2]
        else:
            idx1 = _sample_python(cells1, size=count, rng=rng, replace=(count > len(cells1)))
            idx2 = _sample_python(cells2, size=count, rng=rng, replace=(count > len(cells2)))
        
        pairs = np.column_stack((idx1, idx2))
        all_pairs.append(pairs)
        
        origin_label = f"{c1}+{c2}"
        all_origins.extend([origin_label] * count)
        
    if len(all_pairs) > 0:
        pairs_arr = np.vstack(all_pairs)
        origins_arr = np.array(all_origins, dtype=object)

        # Debug: dump pre-dedup pairs
        if debug_outdir is not None:
            try:
                import os
                os.makedirs(debug_outdir, exist_ok=True)
                pd.DataFrame(pairs_arr, columns=['i1','i2']).to_csv(os.path.join(debug_outdir, 'get_cell_pairs_pre_dedupe.csv'), index=False)
            except Exception:
                pass

        # Remove duplicated rows in the same order as R's duplicated(ca)
        ca_df = pd.DataFrame({
            'cell1': pairs_arr[:, 0],
            'cell2': pairs_arr[:, 1],
            'orig.clusters': origins_arr,
        })
        keep_mask = ~ca_df.duplicated(keep='first')
        pairs_unique = pairs_arr[keep_mask.to_numpy()]
        origins_unique = origins_arr[keep_mask.to_numpy()].tolist()

        # Debug: dump post-dedup pairs
        if debug_outdir is not None:
            try:
                import os
                pd.DataFrame(pairs_unique, columns=['i1','i2']).to_csv(os.path.join(debug_outdir, 'get_cell_pairs_post_dedupe.csv'), index=False)
                pd.DataFrame({'origin': origins_unique}).to_csv(os.path.join(debug_outdir, 'get_cell_pairs_post_origins.csv'), index=False)
            except Exception:
                pass

        return pairs_unique, origins_unique
    else:
        return np.empty((0, 2), dtype=int), []

def _get_meta_cells(X, clusters, n_meta_cells=20, meta_cell_size=20, rng=None):
    """
    Creates within-cluster meta-cells from a count matrix.
    Returns (meta_X, meta_clusters)
    """
    rng = _coerce_rng(rng=rng)
    sample_rng = rng
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters)
    
    meta_vectors = []
    meta_cluster_labels = []
    
    indices = np.arange(X.shape[0])
    
    for c in unique_clusters:
        c_indices = indices[clusters == c]
        n_c = len(c_indices)
        if n_c == 0: continue
        
        # Determine size to sample
        size_to_sample = min(int(np.ceil(0.6 * n_c)), meta_cell_size)
        if size_to_sample < 1: size_to_sample = 1
        
        # R logic: sample(...) replace=FALSE
        for _ in range(n_meta_cells):
            sample_idx = sample_int_r(len(c_indices), size_to_sample, replace=False, rng=sample_rng) - 1
            sample_idx = c_indices[sample_idx]
            
            # Use X[sample_idx] and Compute Mean
            # mean(axis=0) creates a matrix (1, n_genes) -> convert to 1D array
            # R: Matrix::rowMeans(x[,y]) -> This is mean of COLUMNS (cells) in R because X is features x cells
            # In Python X is cells x features. So we want mean across rows (axis=0) to get 1 feature vector.
            
            mean_vec = np.array(X[sample_idx].mean(axis=0)).flatten()
            meta_vectors.append(mean_vec)
            meta_cluster_labels.append(c)
            
    if not meta_vectors:
        return None, []
        
    # Stack -> (N_meta, N_genes)
    meta_X = sp.csr_matrix(np.vstack(meta_vectors))
    return meta_X, np.array(meta_cluster_labels)

def get_artificial_doublets(X, n=3000, clusters=None, resamp=0.25,
                            half_size=0.25, adjust_size=0.25, prop_random=0.1,
                            sel_mode="proportional", n_meta_cells=2, meta_triplets=True, 
                            trim_q=(0.05, 0.95), random_state=0, rng=None, debug_outdir=None):
    """
    Main orchestration function to get artificial doublets.
    
    This corresponds to `getArtificialDoublets` in R.
    """
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    rng = _coerce_rng(random_state=random_state, rng=rng)
    sample_rng = rng
        
    n_cells = X.shape[0]
    ls = np.array(X.sum(axis=1)).flatten()
    
    # 1. Filter cells (Quality Control for Doublet Generation)
    if clusters is None:
        q_min = np.quantile(ls, min(trim_q))
        q_max = np.quantile(ls, max(trim_q))
        w = np.where((ls > 0) & (ls >= q_min) & (ls <= q_max))[0]
    else:
        clusters = np.asarray(clusters)
        w_mask = np.zeros(n_cells, dtype=bool)
        unique_clusters = np.unique(clusters)
        for c in unique_clusters:
            c_mask = (clusters == c)
            ls_c = ls[c_mask]
            if len(ls_c) < 10:
                bounds = (0, np.max(ls_c))
            else:
                bounds = np.quantile(ls_c, sorted(trim_q))
            term = c_mask & (ls > 0) & (ls >= bounds[0]) & (ls <= bounds[1])
            w_mask |= term
        w = np.where(w_mask)[0]
        
    X_curr = X[w]
    if clusters is not None:
        clusters_curr = clusters[w]
    else:
        clusters_curr = None
        
    n_curr = X_curr.shape[0]
    
    # Storage for results
    results_X = []
    results_origins = []

    # 2. Random Doublets
    n_random = int(np.ceil(n * prop_random)) if clusters is not None else n
    
    if n_random > 0:
        # Generate random pairs.
        # Match R: sample 2*n indices, without replacement when possible, then reshape.
        replace = (2 * n_random >= n_curr)
        sampled = sample_int_r(n_curr, size=2 * n_random, replace=replace, rng=sample_rng) - 1
        # R fills sampled values column-major when reshaping into pairs
        pairs_r = sampled.reshape(-1, 2, order='F')
        # Remove self-pairs
        pairs_r = pairs_r[pairs_r[:, 0] != pairs_r[:, 1]]
        
        if clusters is None:
            # If no clusters, everything is random
            X_rand = create_doublets(X_curr, pairs_r, adjust_size=False, resamp=resamp, half_size=half_size, random_state=random_state, rng=rng, debug_outdir=debug_outdir)
            orig_rand = [np.nan] * X_rand.shape[0]
        else:
            # Map random pairs to their clusters
            c1 = clusters_curr[pairs_r[:, 0]]
            c2 = clusters_curr[pairs_r[:, 1]]
            # Match R's paste ordering for random pairs (reverse order)
            orig_rand = [f"{b}+{a}" for a, b in zip(c1, c2)]
            X_rand = create_doublets(X_curr, pairs_r, clusters=clusters_curr, adjust_size=adjust_size, 
                                     resamp=resamp, half_size=half_size, random_state=random_state, rng=rng, debug_outdir=debug_outdir)
        
        results_X.append(X_rand)
        results_origins.extend(orig_rand)

    # 3. Cross-Cluster Doublets (Heterotypic)
    n_main = int(np.ceil(n * (1 - prop_random))) if clusters is not None else n
    n_cluster_dbl = int(np.ceil(n_main * (0.9 if (clusters is not None and n_meta_cells > 0) else 1.0)))
    if n_cluster_dbl > 0 and clusters is not None:
        
        ca_pairs, ca_origins = get_cell_pairs(clusters_curr, n=n_cluster_dbl, sel_mode=sel_mode, rng=rng, debug_outdir=debug_outdir, use_sample_int=True)
        
        if len(ca_pairs) > 0:
            X_cluster = create_doublets(X_curr, ca_pairs, clusters=clusters_curr,
                                        adjust_size=adjust_size, half_size=half_size, resamp=resamp, random_state=random_state, rng=rng, debug_outdir=debug_outdir)
            results_X.append(X_cluster)
            results_origins.extend(ca_origins)

    # 4. Meta Cells
    # Check if we should generate meta cells
    if clusters is not None and len(np.unique(clusters_curr)) < 3: 
        n_meta_cells = 0
        
    if clusters is not None and n_meta_cells > 0:
        # Create doublets from meta cells
        meta_X, meta_cl = _get_meta_cells(X_curr, clusters_curr, n_meta_cells=n_meta_cells, meta_cell_size=30, rng=rng)
        
        if meta_X is not None:
            n_meta_dbl = int(np.ceil(n_main * 0.1)) # 10% of the reduced main count

            ca_meta_pairs, ca_meta_origins = get_cell_pairs(meta_cl, n=n_meta_dbl, rng=rng, debug_outdir=debug_outdir, use_sample_int=True)

            if len(ca_meta_pairs) > 0:
                # R: createDoublets(..., resamp=TRUE) -> resamp=1.0
                X_meta_dbl = create_doublets(meta_X, ca_meta_pairs, clusters=meta_cl,
                                             adjust_size=False, half_size=False, resamp=1.0, random_state=random_state, rng=rng, debug_outdir=debug_outdir)

                results_X.append(X_meta_dbl)
                results_origins.extend(ca_meta_origins)
                 
    # 5. Triplets from Meta Cells
    # R logic: check if clusters have >10% of cells
    # If yes, use those. Else use 3 largest.
    if clusters is not None and meta_triplets:
         unique_c, counts = np.unique(clusters_curr, return_counts=True)
         pct_10 = len(clusters_curr) / 10.0
         large_clusters = unique_c[counts >= pct_10]
         
         if len(large_clusters) > 2:
             target_cl = large_clusters
         else:
             # Sort by counts descending
             sorted_idx = np.argsort(-counts)
             target_cl = unique_c[sorted_idx][:3]
             
         # Mask for cells in target clusters
         w_triplet = np.isin(clusters_curr, target_cl)
         
         if np.sum(w_triplet) > 0:
             # Create meta cells for triplets
             # R: n.meta.cells=1, meta.cell.size=100
             meta_tri_X, meta_tri_cl = _get_meta_cells(X_curr[w_triplet], clusters_curr[w_triplet], 
                                                        n_meta_cells=1, meta_cell_size=100)
             
             if meta_tri_X is not None and meta_tri_X.shape[0] >= 3:
                 # Generate all combinations of length 3
                 # R: expand.grid(i, i, i) -> ca[ca[,1]<ca[,2] & ca[,2]<ca[,3],]
                 # Python: itertools.combinations
                 
                 n_meta_total = meta_tri_X.shape[0]
                 triplet_indices = list(itertools.combinations(range(n_meta_total), 3))
                 
                 if triplet_indices:
                     triplet_indices = np.array(triplet_indices)
                     
                     # Simple average of 3 cells: (A+B+C)/2 -> round
                     # meta_tri_X is sparse
                     
                     X_A = meta_tri_X[triplet_indices[:, 0]]
                     X_B = meta_tri_X[triplet_indices[:, 1]]
                     X_C = meta_tri_X[triplet_indices[:, 2]]
                     
                     X_triplets = (X_A + X_B + X_C) / 2.0
                     if sp.issparse(X_triplets):
                         X_triplets.data = np.round(X_triplets.data)
                     else:
                         X_triplets = np.round(X_triplets)
                     
                     results_X.append(X_triplets)
                     
                     # Origins
                     # R: paste("artTriplet.", ...)
                     # We'll just label them as triplets
                     results_origins.extend([f"artTriplet" for _ in range(X_triplets.shape[0])])

    # Combine results
    if not results_X:
        return {"counts": sp.csr_matrix((0, X.shape[1])), "origins": []}
        
    final_X = sp.vstack(results_X)
    return {"counts": final_X, "origins": results_origins}


def add_doublets_to_adata(adata, clusters_col=None, n_doublets=None, verbose=True):
    """
    High-level wrapper to add artificial doublets to an AnnData object.
    
    This closely mimics `addDoublets` in R, utilizing Scanpy for dependencies.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    clusters_col : str, optional
        Name of column in `adata.obs` containing cluster labels. 
        If None, will look for 'clusters' or run simple clustering.
    n_doublets : int, optional
        Number of doublets to generate. Default is ~20% of cells if not specified.
    
    Returns
    -------
    AnnData
        A new AnnData object containing both Singlets and Doublets.
        The `.obs` dataframe will have a 'type' column ('singlet', 'doublet').
    """
    
    if n_doublets is None:
        n_doublets = int(adata.n_obs * 0.2) # Default heuristic

    # 1. Check for clusters
    clusters = None
    if clusters_col is not None:
        if clusters_col in adata.obs:
            clusters = adata.obs[clusters_col].values
        else:
            if verbose: print(f"Column '{clusters_col}' not found in obs.")
            
    if clusters is None:
        if clusters_col is not None and verbose:
             print(f"Column '{clusters_col}' not found in obs. Doublets will be generated randomly without cluster structure.")
        elif verbose:
             print("No clusters provided. Doublets will be generated randomly without cluster structure.")
            
    if verbose: print(f"Generating {n_doublets} artificial doublets...")
    
    # .X might be dense or sparse
    X = adata.X
    res = get_artificial_doublets(X, n=n_doublets, clusters=clusters)
    
    min_len = min(res["counts"].shape[0], len(res["origins"]))
    
    # 3. Create new AnnData with combined data
    # Create doublet AnnData
    X_dbl = res["counts"]
    obs_dbl = pd.DataFrame(index=[f"art_dbl_{i}" for i in range(X_dbl.shape[0])])
    obs_dbl["type"] = "doublet"
    obs_dbl["origin"] = res["origins"]
    
    adata_dbl = AnnData(X=X_dbl, obs=obs_dbl)
    adata_dbl.var = adata.var.copy() # Share gene names
    
    # Prepare original AnnData
    adata_singlets = adata.copy()
    adata_singlets.obs["type"] = "singlet"
    adata_singlets.obs["origin"] = clusters if clusters is not None else "real"
    
    # Concatenate
    # This matches the behavior of SingleCellExperiment::cbind
    return adata_singlets.concatenate(adata_dbl, index_unique=None)
