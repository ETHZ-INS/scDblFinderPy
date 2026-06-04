import numpy as np


def kmknn_indices(X, k):
    """Compute exact k-nearest neighbor indices using pairwise Euclidean distances.

    Pure-Python implementation (numpy) that returns a (n_samples, k) array of
    0-based neighbor indices. Deterministic and exact; suitable as a drop-in
    replacement for BiocNeighbors::findKNN(..., BNPARAM=KmknnParam()).
    Note: this is O(n^2) in memory/time and intended for moderate-sized inputs
    such as centroid matrices. For large datasets use an approximate method.
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    if k >= n_samples:
        # return all other indices (exclude self)
        idx = np.tile(np.arange(n_samples), (n_samples, 1))
        return idx[:, :k]

    # Efficient squared Euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms = np.sum(X * X, axis=1)
    # compute Gram matrix
    gram = X.dot(X.T)
    # distances squared
    d2 = norms[:, None] + norms[None, :] - 2.0 * gram
    # numerical safety
    d2 = np.maximum(d2, 0.0)

    # set self-distance to +inf so self is not selected
    np.fill_diagonal(d2, np.inf)

    # use argpartition for efficiency then sort the k candidates
    idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
    # For each row, sort the k indices by distance
    row_idx = np.arange(n_samples)[:, None]
    sorted_order = np.argsort(d2[row_idx, idx], axis=1)
    knn_idx = idx[row_idx, sorted_order]
    # knn_idx now has shape (n_samples, k)
    return knn_idx.reshape(n_samples, k)


__all__ = ["kmknn_indices"]
