import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def make_knn_graph_bluster(X, k=10, metric='euclidean', mode='jaccard', mutual=True):
    """Construct a KNN graph similar to bluster::makeKNNGraph.

    Steps:
    - Compute the k nearest neighbors for each point (excluding self).
    - Compute an SNN weight between two points as the Jaccard index of their
      neighbor sets (|N(i) \cap N(j)| / |N(i) \cup N(j)|) by default.
    - Return a symmetric sparse connectivity matrix of shape (n,n) with
      weights in [0,1].

    This is a pragmatic Python reimplementation intended to reproduce R's
    SNN-style graph used by `fastcluster`.
    """
    X_arr = np.asarray(X)
    n_samples = X_arr.shape[0]

    if k <= 0:
        raise ValueError("k must be positive")

    # For faithful Python-only behavior use exact kmknn indices for moderate sizes
    try:
        from .biocneighbors_kmknn import kmknn_indices
        indices = kmknn_indices(X_arr, k)
    except Exception:
        # Fallback to sklearn NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, n_samples), metric=metric)
        nn.fit(X_arr)
        distances, indices = nn.kneighbors(X_arr)
        # drop self if present
        if indices.shape[1] > 0 and np.all(indices[:, 0] == np.arange(n_samples)):
            indices = indices[:, 1:]
        else:
            indices = indices[:, :k]

    # Drop self if present as first neighbor
    if indices.shape[1] > 0 and np.all(indices[:, 0] == np.arange(n_samples)):
        nbrs = indices[:, 1:]
    else:
        nbrs = indices[:, :k]

    # If 'knn' mode requested, construct simple undirected KNN graph like
    # bluster::makeKNNGraph -> neighborsToKNNGraph: edges (i, j) for any j in N(i).
    if mode == 'knn':
        rows = np.repeat(np.arange(n_samples), nbrs.shape[1])
        cols = nbrs.flatten()
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        # Symmetrize by stacking reversed edges
        rows_sym = np.concatenate([rows, cols])
        cols_sym = np.concatenate([cols, rows])
        data_sym = np.ones_like(rows_sym, dtype=float)
        mat = sp.csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n_samples, n_samples), dtype=float)
        # simplify (duplicates will be summed to >1). Convert to binary adjacency
        mat.data = np.where(mat.data > 0, 1.0, 0.0)
        mat.eliminate_zeros()
        return mat

    # Build SNN weights (fallback modes)
    rows = []
    cols = []
    data = []

    # For efficiency, convert neighbor lists to sets
    nbr_sets = [set(row.tolist()) for row in nbrs]

    for i in range(n_samples):
        Ni = nbr_sets[i]
        for j in nbrs[i]:
            if j <= i:
                continue
            # if mutual is required, ensure i is also among j's neighbors
            if mutual and (i not in nbr_sets[j]):
                continue
            Nj = nbr_sets[j]
            inter = len(Ni.intersection(Nj))
            if inter == 0:
                continue
            if mode == 'jaccard':
                union = len(Ni.union(Nj))
                w = inter / union if union > 0 else 0.0
            else:
                # fallback: count of shared neighbors
                w = float(inter)

            if w > 0:
                rows.append(i)
                cols.append(j)
                data.append(w)

    if len(data) == 0:
        return sp.csr_matrix((n_samples, n_samples), dtype=float)

    # Symmetrize
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])

    mat = sp.csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n_samples, n_samples), dtype=float)

    return mat
