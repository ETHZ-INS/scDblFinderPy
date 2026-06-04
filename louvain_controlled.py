from __future__ import annotations

import random

import numpy as np
import scipy.sparse as sp

from .rng import coerce_rng


def _as_csr(connectivities):
    if sp.issparse(connectivities):
        return connectivities.tocsr().astype(float)
    return sp.csr_matrix(np.asarray(connectivities, dtype=float))


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(np.asarray(labels, dtype=int), return_inverse=True)
    return inverse.astype(int, copy=False)


def _modularity(connectivities: sp.csr_matrix, labels: np.ndarray, resolution: float = 1.0) -> float:
    labels = np.asarray(labels, dtype=int)
    total = float(connectivities.sum())
    if total <= 0:
        return 0.0

    degrees = np.asarray(connectivities.sum(axis=1)).ravel()
    score = 0.0
    for community in np.unique(labels):
        mask = labels == community
        if not np.any(mask):
            continue
        internal = float(connectivities[mask][:, mask].sum())
        degree_sum = float(degrees[mask].sum())
        score += internal / total - resolution * (degree_sum / total) ** 2
    return float(score)


def _candidate_communities(connectivities: sp.csr_matrix, labels: np.ndarray, node: int) -> np.ndarray:
    start = connectivities.indptr[node]
    end = connectivities.indptr[node + 1]
    neighbors = connectivities.indices[start:end]
    if neighbors.size == 0:
        return np.array([int(labels[node])], dtype=int)
    candidates = np.unique(labels[neighbors].astype(int))
    current = int(labels[node])
    if current not in candidates:
        candidates = np.append(candidates, current)
    return np.unique(candidates).astype(int)


def _local_moving_phase(connectivities: sp.csr_matrix, rng, resolution: float = 1.0, max_passes: int = 100, tol: float = 1e-12):
    n_vertices = connectivities.shape[0]
    labels = np.arange(n_vertices, dtype=int)
    current_modularity = _modularity(connectivities, labels, resolution=resolution)

    for _ in range(max_passes):
        moved = False
        order = np.asarray(rng.permutation(n_vertices), dtype=int)

        for node in order:
            current_label = int(labels[node])
            best_label = current_label
            best_modularity = current_modularity
            candidates = _candidate_communities(connectivities, labels, int(node))

            for candidate in candidates:
                if candidate == current_label:
                    continue
                trial = labels.copy()
                trial[node] = int(candidate)
                trial = _relabel_contiguous(trial)
                modularity = _modularity(connectivities, trial, resolution=resolution)
                if modularity > best_modularity + tol or (abs(modularity - best_modularity) <= tol and int(candidate) < int(best_label)):
                    best_modularity = modularity
                    best_label = int(candidate)

            if best_label != current_label:
                labels[node] = best_label
                labels = _relabel_contiguous(labels)
                current_modularity = best_modularity
                moved = True

        if not moved:
            break

    return _relabel_contiguous(labels), float(current_modularity)


def _aggregate_graph(connectivities: sp.csr_matrix, labels: np.ndarray):
    labels = _relabel_contiguous(labels)
    n_vertices = connectivities.shape[0]
    n_communities = int(labels.max()) + 1 if labels.size else 0
    if n_communities == 0:
        return sp.csr_matrix((0, 0), dtype=float)
    membership = sp.csr_matrix((np.ones(n_vertices, dtype=float), (np.arange(n_vertices), labels)), shape=(n_vertices, n_communities))
    aggregated = membership.T @ connectivities @ membership
    if sp.issparse(aggregated):
        aggregated = aggregated.tocsr().astype(float)
        aggregated.eliminate_zeros()
    else:
        aggregated = sp.csr_matrix(np.asarray(aggregated, dtype=float))
    return aggregated


def controlled_louvain_labels(connectivities, random_state=0, max_levels: int = 20, max_passes: int = 100, resolution: float = 1.0):
    """Deterministic Louvain-style community detection with explicit RNG control.

    The algorithm uses:
    - a controllable node visitation order driven by ``random_state``;
    - smallest-community-id tie-breaking when modularity gains are equal;
    - repeated aggregation passes until no further improvement is possible.
    """
    rng = coerce_rng(random_state=random_state, rng=random_state)
    current_graph = _as_csr(connectivities)
    if current_graph.shape[0] == 0:
        return np.array([], dtype=int)

    try:
        import igraph as ig

        ig.set_random_number_generator(random.Random(int(random_state)))
        coo = current_graph.tocoo()
        mask = coo.row < coo.col
        edges = list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))
        weights = coo.data[mask].astype(float).tolist()
        graph = ig.Graph(n=current_graph.shape[0], edges=edges, directed=False)
        if len(weights) == len(edges):
            graph.es["weight"] = weights
            membership = graph.community_multilevel(weights="weight").membership
        else:
            membership = graph.community_multilevel().membership
        return _relabel_contiguous(np.asarray(membership, dtype=int))
    except Exception:
        pass

    membership = np.arange(current_graph.shape[0], dtype=int)

    for _ in range(max_levels):
        level_labels, _ = _local_moving_phase(current_graph, rng, resolution=resolution, max_passes=max_passes)
        level_labels = _relabel_contiguous(level_labels)
        membership = level_labels[membership]

        if int(level_labels.max()) + 1 >= current_graph.shape[0]:
            break

        aggregated = _aggregate_graph(current_graph, level_labels)
        if aggregated.shape[0] == current_graph.shape[0]:
            break
        current_graph = aggregated

        if current_graph.shape[0] <= 1:
            break

    return _relabel_contiguous(membership)
