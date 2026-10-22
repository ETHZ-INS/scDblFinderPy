import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def _find_knn(coords, k):
    """
    True (non-self) k-nearest-neighbor search over `coords`, mirroring the
    exclude-self convention of R's `BiocNeighbors::findKNN` (and matching the
    self-masking approach already used for KNN features in scDblFinder.py).

    Returns (index, distance), each of shape (n_obs, k_eff) with
    k_eff = min(k, n_obs - 1).
    """
    n_obs = coords.shape[0]
    k_eff = min(k, n_obs - 1)
    n_query = min(k_eff + 1, n_obs)

    nn = NearestNeighbors(n_neighbors=n_query).fit(coords)
    dist, idx = nn.kneighbors(coords)

    self_mask = idx == np.arange(n_obs)[:, None]
    dist = np.where(self_mask, np.inf, dist)
    order = np.argsort(dist, axis=1, kind='stable')
    idx = np.take_along_axis(idx, order, axis=1)[:, :k_eff]
    dist = np.take_along_axis(dist, order, axis=1)[:, :k_eff]
    return idx, dist


def smooth_doublet_scores(
    x,
    knn=None,
    coords="X_pca",
    k=20,
    alpha=0.5,
    gamma=1,
    weights=True,
    decay_by_distance="linear",
    score_column="scDblFinder_score",
    out_column="smoothedDoubletScore",
):
    """
    Apply distance-weighted kNN smoothing to per-cell doublet probabilities.

    Ports R's `smoothDoubletScores()` (scDblFinder/R/smoothScores.R). Blends
    each cell's own doublet score, in logit space, with a distance-weighted
    consensus of its neighborhood, optionally raised to a power (`gamma`) to
    amplify coherent high-probability neighborhoods while suppressing
    isolated noisy predictions. This is a standalone post-processing step:
    it is not called automatically by `compute_doublet_score`, so call it
    explicitly afterwards on cells that have a doublet score and some
    embedding (e.g. PCA) to smooth over.

    Parameters
    ----------
    x : AnnData or array-like
        Either an AnnData object with a `score_column` column in `.obs`, or
        a 1D array-like of doublet probabilities in [0, 1].
    knn : dict, optional
        Precomputed neighbors as a dict with 'index' and 'distance' arrays,
        each of shape (n_cells, k) (e.g. as stored in
        `adata.obsm['knn_indices']` / `adata.obsm['knn_distances']` by
        `compute_doublet_score`'s internal KNN step, when its neighbor
        definition is the one desired here). If None, neighbors are computed
        from `coords`.
    coords : str or array-like
        Either the name of an `.obsm` entry in `x` to use for the KNN search
        (only valid when `x` is an AnnData; default 'X_pca'), or a
        (n_cells, n_dims) coordinate matrix. Ignored if `knn` is given.
    k : int
        Number of nearest neighbors to compute if `knn` is None. Capped at
        `n_cells - 1`. Default 20.
    alpha : float
        Blending factor in [0, 1] between the cell's own score (0) and the
        neighborhood consensus (1). Default 0.5.
    gamma : float
        Non-linear amplification exponent (>= 1). Values above 1
        exponentially amplify clusters of high-probability cells. Default 1
        (no amplification).
    weights : bool
        If True (default), use adaptive Gaussian kernel weights based on
        continuous distance. If False, use uniform weights (1/k) across all
        neighbors.
    decay_by_distance : {'linear', 'exp', 'none'}
        Whether (and how) to shrink the neighborhood's influence for
        isolated cells, based on their distance to their first neighbor
        relative to the dataset's median first-neighbor distance, forcing
        them to rely mostly on their own raw score. Default 'linear'.
    score_column : str
        Column in `x.obs` holding the raw doublet scores, if `x` is an
        AnnData. Default 'scDblFinder_score' (matches
        `compute_doublet_score`'s output column).
    out_column : str
        Column to store the smoothed scores in, if `x` is an AnnData.
        Default 'smoothedDoubletScore'.

    Returns
    -------
    AnnData or numpy.ndarray
        If `x` is an AnnData, returns it with `out_column` added to `.obs`.
        If `x` is array-like, returns a numpy array of smoothed scores.
    """
    if decay_by_distance not in ("linear", "exp", "none"):
        raise ValueError("decay_by_distance must be one of 'linear', 'exp', 'none'")

    is_adata = isinstance(x, AnnData)
    if is_adata:
        if score_column not in x.obs.columns:
            raise ValueError(f"'{score_column}' not found in adata.obs")
        probs = np.asarray(x.obs[score_column].values, dtype=float)
        if knn is None and isinstance(coords, str):
            if coords not in x.obsm:
                raise ValueError(f"'{coords}' not found in adata.obsm")
            coords = np.asarray(x.obsm[coords])
    else:
        probs = np.asarray(x, dtype=float)
        if knn is None and (coords is None or np.ndim(coords) != 2):
            raise ValueError(
                "If 'x' is not an AnnData, provide either 'knn' or a 2D "
                "coordinates matrix as 'coords'."
            )

    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("All scores in 'x' must lie in [0, 1].")

    if knn is None:
        idx, dist = _find_knn(np.asarray(coords, dtype=float), k)
    else:
        idx = np.asarray(knn["index"])
        dist = np.asarray(knn["distance"])

    probs_safe = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(probs_safe / (1 - probs_safe))
    neighbor_logits = logits[idx]

    if weights:
        pos_dist = dist[dist > 0]
        min_dist = (pos_dist.min() / 10.0) if pos_dist.size > 0 else 1e-6
        sigma = np.median(dist, axis=1) + min_dist
        w = np.exp(-(dist ** 2) / (2 * (sigma[:, None] ** 2)))
        row_sums = w.sum(axis=1, keepdims=True)
        w = w / np.where(row_sums == 0, 1.0, row_sums)
    else:
        w = np.full_like(dist, 1.0 / dist.shape[1], dtype=float)

    if gamma == 1:
        weighted_consensus = np.sum(w * neighbor_logits, axis=1)
    else:
        signed_power_logits = np.sign(neighbor_logits) * (np.abs(neighbor_logits) ** gamma)
        weighted_consensus = np.sum(w * signed_power_logits, axis=1)
        weighted_consensus = np.sign(weighted_consensus) * (np.abs(weighted_consensus) ** (1.0 / gamma))

    if decay_by_distance != "none":
        first_dist = dist[:, 0]
        med_first_dist = np.median(first_dist) + 1e-8
        if decay_by_distance == "linear":
            cell_alpha = alpha * np.minimum(first_dist / med_first_dist, 1.0)
        else:
            cell_alpha = alpha * np.exp(-(first_dist / med_first_dist))
    else:
        cell_alpha = alpha

    final_logits = (1 - cell_alpha) * logits + cell_alpha * weighted_consensus
    smoothed_probs = 1.0 / (1.0 + np.exp(-final_logits))

    if not is_adata:
        return smoothed_probs

    x.obs[out_column] = smoothed_probs
    return x
