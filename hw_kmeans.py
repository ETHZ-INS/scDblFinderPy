import numpy as np
from .rng import coerce_rng
import os
import sys
from multiprocessing import get_context
from multiprocessing import shared_memory

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


def _parallel_start_worker(args):
    name, shape, dtype_str, init_idx_arr, max_iter_local, n_clusters_local = args
    shm_local = shared_memory.SharedMemory(name=name)
    try:
        X_local = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm_local.buf)
        centers, inertia = _single_start_run_shared(X_local, init_idx_arr, max_iter_local, n_clusters_local)
    finally:
        shm_local.close()
    return centers, inertia


def _single_start_run_shared(X_local, init_idx_arr, max_iter, n_clusters):
    n_samples = X_local.shape[0]
    centers = X_local[init_idx_arr].copy()
    x_norm_sq = np.einsum("ij,ij->i", X_local, X_local)
    center_norm_sq = np.einsum("ij,ij->i", centers, centers)
    d2 = x_norm_sq[:, None] + center_norm_sq[None, :] - 2.0 * (X_local @ centers.T)
    labels = np.argmin(d2, axis=1)

    for it in range(int(max_iter)):
        centers, sums, counts, sums_sq = _compute_centers_sums_counts(X_local, labels, n_clusters)
        sum_norm_sq = np.einsum("ij,ij->i", sums, sums)
        cluster_sse = np.zeros_like(sums_sq)
        np.divide(sum_norm_sq, counts, out=cluster_sse, where=counts > 0)
        cluster_sse = sums_sq - cluster_sse
        moved = False

        for i in range(n_samples):
            ci = labels[i]
            xi = X_local[i]
            xi_norm_sq = x_norm_sq[i]
            dot_sums_xi = sums @ xi

            # SSE if xi is added to each cluster j.
            new_cluster_sum_norm_sq = sum_norm_sq + 2.0 * dot_sums_xi + xi_norm_sq
            new_j_sse = (sums_sq + xi_norm_sq) - (new_cluster_sum_norm_sq / (counts + 1))

            # SSE if xi is removed from its current cluster ci.
            if counts[ci] > 1:
                ci_sum_norm_sq = sum_norm_sq[ci] - 2.0 * dot_sums_xi[ci] + xi_norm_sq
                new_i_sse = (sums_sq[ci] - xi_norm_sq) - (ci_sum_norm_sq / (counts[ci] - 1))
            else:
                new_i_sse = 0.0

            delta = (new_j_sse - cluster_sse) + (new_i_sse - cluster_sse[ci])
            delta[ci] = 0.0
            best_j = int(np.argmin(delta))
            best_delta = float(delta[best_j])

            if best_j != ci:
                labels[i] = best_j
                counts[ci] -= 1
                counts[best_j] += 1
                sums[ci] -= xi
                sums[best_j] += xi
                sums_sq[ci] -= xi_norm_sq
                sums_sq[best_j] += xi_norm_sq
                sum_norm_sq[ci] = np.dot(sums[ci], sums[ci])
                sum_norm_sq[best_j] = np.dot(sums[best_j], sums[best_j])
                if counts[ci] > 0:
                    cluster_sse[ci] = sums_sq[ci] - (sum_norm_sq[ci] / counts[ci])
                else:
                    cluster_sse[ci] = 0.0
                cluster_sse[best_j] = sums_sq[best_j] - (sum_norm_sq[best_j] / counts[best_j])
                centers[ci] = sums[ci] / counts[ci] if counts[ci] > 0 else centers[ci]
                centers[best_j] = sums[best_j] / counts[best_j]
                moved = True

        if np.any(counts == 0):
            labels, centers, counts = _reseed_empty_clusters(X_local, labels, centers, counts)
            moved = True

        if not moved:
            break

    inertia = np.sum((X_local - centers[labels])**2)
    return centers, float(inertia)


def _parallel_spawn_safe():
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    return bool(main_file) and os.path.exists(main_file)


def _emit_trace(trace, event, **payload):
    if trace is None:
        return
    record = {"event": event, **payload}
    if callable(trace):
        trace(record)
    else:
        trace.append(record)


def _squared_distance(a, b):
    diff = a - b
    return float(np.dot(diff, diff))


def _reseed_empty_clusters(X, labels, centers, counts):
    """Deterministically reseed any empty clusters.

    We move the point with the largest squared error to its current center
    into each empty cluster, choosing donor points in a stable order.
    """
    empty_clusters = np.where(counts == 0)[0]
    if empty_clusters.size == 0:
        return labels, centers, counts

    for empty_cluster in empty_clusters:
        occupied = np.where(counts > 1)[0]
        if occupied.size == 0:
            break

        best_point = None
        best_cluster = None
        best_error = -np.inf

        for cluster_idx in occupied:
            members = np.where(labels == cluster_idx)[0]
            if members.size == 0:
                continue
            errors = np.sum((X[members] - centers[cluster_idx]) ** 2, axis=1)
            local_pos = int(np.argmax(errors))
            candidate_point = int(members[local_pos])
            candidate_error = float(errors[local_pos])
            if candidate_error > best_error or (candidate_error == best_error and candidate_point < best_point):
                best_error = candidate_error
                best_point = candidate_point
                best_cluster = cluster_idx

        if best_point is None:
            break

        xi = X[best_point]
        labels[best_point] = empty_cluster
        counts[best_cluster] -= 1
        counts[empty_cluster] += 1
        centers[best_cluster] = X[labels == best_cluster].mean(axis=0)
        centers[empty_cluster] = xi

    return labels, centers, counts


def _compute_centers_sums_counts(X, labels, k):
    n_features = X.shape[1]
    sums = np.zeros((k, n_features), dtype=float)
    counts = np.zeros(k, dtype=int)
    sums_sq = np.zeros(k, dtype=float)
    for j in range(k):
        mask = labels == j
        counts[j] = np.count_nonzero(mask)
        if counts[j] > 0:
            sums[j] = X[mask].sum(axis=0)
            # sum of squared norms for cluster j
            sums_sq[j] = np.sum((X[mask] ** 2).sum(axis=1))
    centers = np.zeros_like(sums)
    nonzero = counts > 0
    centers[nonzero] = sums[nonzero] / counts[nonzero][:, None]
    return centers, sums, counts, sums_sq


def _assign_two_nearest(X, centers):
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    ic1 = np.empty(n_samples, dtype=int)
    ic2 = np.empty(n_samples, dtype=int)
    d1 = np.empty(n_samples, dtype=float)
    d2 = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        best1 = 0
        best2 = 1 if n_clusters > 1 else 0
        dist1 = _squared_distance(X[i], centers[0])
        dist2 = _squared_distance(X[i], centers[1]) if n_clusters > 1 else dist1
        if dist2 < dist1:
            best1, best2 = 1, 0
            dist1, dist2 = dist2, dist1

        for cluster_idx in range(2, n_clusters):
            dist = _squared_distance(X[i], centers[cluster_idx])
            if dist < dist1:
                dist2, best2 = dist1, best1
                dist1, best1 = dist, cluster_idx
            elif dist < dist2:
                dist2, best2 = dist, cluster_idx

        ic1[i] = best1
        ic2[i] = best2
        d1[i] = dist1
        d2[i] = dist2

    return ic1, ic2, d1, d2


def _init_cluster_stats(X, labels, n_clusters):
    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features), dtype=float)
    counts = np.zeros(n_clusters, dtype=int)
    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        counts[cluster_idx] = int(np.count_nonzero(mask))
        if counts[cluster_idx] > 0:
            centers[cluster_idx] = X[mask].mean(axis=0)
    return centers, counts


if njit is not None:

    @njit(cache=True)
    def _squared_distance_nb(a, b):
        total = 0.0
        for j in range(a.shape[0]):
            diff = a[j] - b[j]
            total += diff * diff
        return total

    @njit(cache=True)
    def _assign_two_nearest_nb(X, centers):
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        ic1 = np.empty(n_samples, dtype=np.int64)
        ic2 = np.empty(n_samples, dtype=np.int64)
        d1 = np.empty(n_samples, dtype=np.float64)
        d2 = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            best1 = 0
            best2 = 0 if n_clusters == 1 else 1
            dist1 = _squared_distance_nb(X[i], centers[0])
            dist2 = dist1 if n_clusters == 1 else _squared_distance_nb(X[i], centers[1])
            if n_clusters > 1 and dist2 < dist1:
                best1 = 1
                best2 = 0
                tmp = dist1
                dist1 = dist2
                dist2 = tmp

            for cluster_idx in range(2, n_clusters):
                dist = _squared_distance_nb(X[i], centers[cluster_idx])
                if dist < dist1:
                    dist2 = dist1
                    best2 = best1
                    dist1 = dist
                    best1 = cluster_idx
                elif dist < dist2:
                    dist2 = dist
                    best2 = cluster_idx

            ic1[i] = best1
            ic2[i] = best2
            d1[i] = dist1
            d2[i] = dist2

        return ic1, ic2, d1, d2

    @njit(cache=True)
    def _init_cluster_stats_nb(X, labels, n_clusters):
        n_samples, n_features = X.shape
        centers = np.zeros((n_clusters, n_features), dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.int64)
        for i in range(n_samples):
            cluster_idx = labels[i]
            counts[cluster_idx] += 1
            for j in range(n_features):
                centers[cluster_idx, j] += X[i, j]
        for cluster_idx in range(n_clusters):
            if counts[cluster_idx] > 0:
                inv = 1.0 / counts[cluster_idx]
                for j in range(n_features):
                    centers[cluster_idx, j] *= inv
        return centers, counts

    @njit(cache=True)
    def _reseed_empty_clusters_nb(X, labels, centers, counts):
        n_samples, n_features = X.shape
        n_clusters = centers.shape[0]
        for empty_cluster in range(n_clusters):
            if counts[empty_cluster] != 0:
                continue

            best_point = -1
            best_cluster = -1
            best_error = -1.0

            for cluster_idx in range(n_clusters):
                if counts[cluster_idx] <= 1:
                    continue
                for i in range(n_samples):
                    if labels[i] != cluster_idx:
                        continue
                    error = 0.0
                    for j in range(n_features):
                        diff = X[i, j] - centers[cluster_idx, j]
                        error += diff * diff
                    if error > best_error or (error == best_error and (best_point == -1 or i < best_point)):
                        best_error = error
                        best_point = i
                        best_cluster = cluster_idx

            if best_point == -1:
                continue

            old_cluster = labels[best_point]
            labels[best_point] = empty_cluster
            counts[old_cluster] -= 1
            counts[empty_cluster] += 1

            for j in range(n_features):
                centers[old_cluster, j] = 0.0
            for i in range(n_samples):
                if labels[i] == old_cluster:
                    for j in range(n_features):
                        centers[old_cluster, j] += X[i, j]
            if counts[old_cluster] > 0:
                inv_old = 1.0 / counts[old_cluster]
                for j in range(n_features):
                    centers[old_cluster, j] *= inv_old

            for j in range(n_features):
                centers[empty_cluster, j] = X[best_point, j]

        return labels, centers, counts

    @njit(cache=True)
    def _hartigan_wong_core_nb(X, init_idx_arr, max_iter, n_clusters):
        n_samples, n_features = X.shape
        centers = np.empty((n_clusters, n_features), dtype=np.float64)
        for cluster_idx in range(n_clusters):
            src = init_idx_arr[cluster_idx]
            for j in range(n_features):
                centers[cluster_idx, j] = X[src, j]

        ic1, ic2, d1, d2 = _assign_two_nearest_nb(X, centers)
        labels = ic1.copy()
        centers, counts = _init_cluster_stats_nb(X, labels, n_clusters)

        if np.any(counts == 0):
            labels, centers, counts = _reseed_empty_clusters_nb(X, labels, centers, counts)
            ic1 = labels.copy()

        big = 1.0e30
        an1 = np.empty(n_clusters, dtype=np.float64)
        an2 = np.empty(n_clusters, dtype=np.float64)
        ncp = np.empty(n_clusters, dtype=np.int64)
        itran = np.ones(n_clusters + 1, dtype=np.int64)
        live = np.zeros(n_clusters, dtype=np.int64)
        for cluster_idx in range(n_clusters):
            ncp[cluster_idx] = -1
            if counts[cluster_idx] > 1:
                an1[cluster_idx] = counts[cluster_idx] / (counts[cluster_idx] - 1.0)
            else:
                an1[cluster_idx] = big
            an2[cluster_idx] = counts[cluster_idx] / (counts[cluster_idx] + 1.0)

        d = d2 * an2[ic2]
        indx = 0
        i_max_qtr = max(1, max_iter) * n_samples

        for _iteration in range(max_iter):
            for cluster_idx in range(n_clusters):
                if itran[cluster_idx] == 1:
                    live[cluster_idx] = n_samples + 1

            for i in range(n_samples):
                indx += 1
                step = i + 1
                l1 = ic1[i]
                l2 = ic2[i]
                ll = l2

                if counts[l1] == 1:
                    continue

                if ncp[l1] != 0:
                    d[i] = _squared_distance_nb(X[i], centers[l1]) * an1[l1]

                r2 = _squared_distance_nb(X[i], centers[l2]) * an2[l2]
                best_l2 = l2
                for cluster_idx in range(n_clusters):
                    if ((step >= live[l1] and step >= live[cluster_idx]) or cluster_idx == l1 or cluster_idx == ll):
                        continue
                    rr = r2 / an2[cluster_idx]
                    dc = _squared_distance_nb(X[i], centers[cluster_idx])
                    if dc < rr:
                        r2 = dc * an2[cluster_idx]
                        best_l2 = cluster_idx

                if r2 >= d[i]:
                    ic2[i] = best_l2
                    continue

                indx = 0
                live[l1] = n_samples + i + 1
                live[best_l2] = n_samples + i + 1
                ncp[l1] = i + 1
                ncp[best_l2] = i + 1

                al1 = float(counts[l1])
                alw = al1 - 1.0
                al2 = float(counts[best_l2])
                alt = al2 + 1.0
                for j in range(n_features):
                    xi = X[i, j]
                    centers[l1, j] = (centers[l1, j] * al1 - xi) / alw
                    centers[best_l2, j] = (centers[best_l2, j] * al2 + xi) / alt
                counts[l1] -= 1
                counts[best_l2] += 1
                an2[l1] = alw / al1
                if alw <= 1.0:
                    an1[l1] = big
                else:
                    an1[l1] = alw / (alw - 1.0)
                an1[best_l2] = alt / al2
                an2[best_l2] = alt / (alt + 1.0)
                ic1[i] = best_l2
                ic2[i] = l1

            if indx == n_samples:
                break

            icount = 0
            istep = 0
            while True:
                for i in range(n_samples):
                    icount += 1
                    istep += 1
                    if istep >= i_max_qtr:
                        inertia = 0.0
                        for i2 in range(n_samples):
                            cluster_idx = ic1[i2]
                            for j in range(n_features):
                                diff = X[i2, j] - centers[cluster_idx, j]
                                inertia += diff * diff
                        return ic1.copy(), centers.copy(), inertia

                    l1 = ic1[i]
                    l2 = ic2[i]

                    if counts[l1] == 1:
                        continue

                    if istep <= ncp[l1]:
                        d[i] = _squared_distance_nb(X[i], centers[l1]) * an1[l1]

                    if not (istep < ncp[l1] or istep < ncp[l2]):
                        continue

                    r2 = d[i] / an2[l2]
                    dd = _squared_distance_nb(X[i], centers[l2])
                    if dd >= r2:
                        continue

                    icount = 0
                    indx = 0
                    itran[l1] = 1
                    itran[l2] = 1
                    ncp[l1] = istep + n_samples
                    ncp[l2] = istep + n_samples

                    al1 = float(counts[l1])
                    alw = al1 - 1.0
                    al2 = float(counts[l2])
                    alt = al2 + 1.0
                    for j in range(n_features):
                        xi = X[i, j]
                        centers[l1, j] = (centers[l1, j] * al1 - xi) / alw
                        centers[l2, j] = (centers[l2, j] * al2 + xi) / alt
                    counts[l1] -= 1
                    counts[l2] += 1
                    an2[l1] = alw / al1
                    if alw <= 1.0:
                        an1[l1] = big
                    else:
                        an1[l1] = alw / (alw - 1.0)
                    an1[l2] = alt / al2
                    an2[l2] = alt / (alt + 1.0)
                    ic1[i] = l2
                    ic2[i] = l1

                if icount == n_samples:
                    break

            for cluster_idx in range(n_clusters):
                itran[cluster_idx] = 0
                live[cluster_idx] -= n_samples

            if np.any(counts == 0):
                labels = ic1.copy()
                labels, centers, counts = _reseed_empty_clusters_nb(X, labels, centers, counts)
                ic1 = labels.copy()
                for cluster_idx in range(n_clusters):
                    if counts[cluster_idx] > 1:
                        an1[cluster_idx] = counts[cluster_idx] / (counts[cluster_idx] - 1.0)
                    else:
                        an1[cluster_idx] = big
                    an2[cluster_idx] = counts[cluster_idx] / (counts[cluster_idx] + 1.0)
                for i in range(n_samples):
                    ic2[i] = 0 if n_clusters == 1 else ic2[i]
                d = d2 * an2[ic2]

            for cluster_idx in range(n_clusters):
                ncp[cluster_idx] = 0

        inertia = 0.0
        for i in range(n_samples):
            cluster_idx = ic1[i]
            for j in range(n_features):
                diff = X[i, j] - centers[cluster_idx, j]
                inertia += diff * diff

        return ic1.copy(), centers.copy(), inertia


def _hartigan_wong_port(X, init_idx_arr, max_iter, n_clusters, trace=None, start=0):
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    centers = X[init_idx_arr].copy()

    ic1, ic2, d1, d2 = _assign_two_nearest(X, centers)
    labels = ic1.copy()
    centers, counts = _init_cluster_stats(X, labels, n_clusters)

    # If any empty clusters after initial assignment, reseed deterministically
    if np.any(counts == 0):
        labels, centers, counts = _reseed_empty_clusters(X, labels, centers, counts)
        ic1 = labels.copy()

    big = 1.0e30
    an1 = np.full(n_clusters, big, dtype=float)
    an2 = np.empty(n_clusters, dtype=float)
    ncp = np.full(n_clusters, -1, dtype=int)
    itran = np.ones(n_clusters + 1, dtype=int)
    live = np.zeros(n_clusters, dtype=int)

    for cluster_idx in range(n_clusters):
        if counts[cluster_idx] > 1:
            an1[cluster_idx] = counts[cluster_idx] / float(counts[cluster_idx] - 1)
        an2[cluster_idx] = counts[cluster_idx] / float(counts[cluster_idx] + 1)

    d = d2 * an2[ic2]
    indx = 0
    i_max_qtr = max(1, int(max_iter)) * n_samples

    for iteration in range(int(max_iter)):
        # OPTRA
        for cluster_idx in range(n_clusters):
            if itran[cluster_idx] == 1:
                live[cluster_idx] = n_samples + 1

        for i in range(n_samples):
            step = i + 1
            indx += 1
            l1 = int(ic1[i])
            l2 = int(ic2[i])
            ll = l2

            if counts[l1] == 1:
                continue

            if ncp[l1] != 0:
                d[i] = _squared_distance(X[i], centers[l1]) * an1[l1]

            r2 = _squared_distance(X[i], centers[l2]) * an2[l2]
            best_l2 = l2
            for cluster_idx in range(n_clusters):
                if ((step >= live[l1] and step >= live[cluster_idx]) or cluster_idx == l1 or cluster_idx == ll):
                    continue
                rr = r2 / an2[cluster_idx]
                dc = 0.0
                for j in range(n_features):
                    dd = X[i, j] - centers[cluster_idx, j]
                    dc += dd * dd
                    if dc >= rr:
                        break
                if dc < rr:
                    r2 = dc * an2[cluster_idx]
                    best_l2 = cluster_idx

            if r2 >= d[i]:
                ic2[i] = best_l2
                continue

            indx = 0
            live[l1] = n_samples + i + 1
            live[best_l2] = n_samples + i + 1
            ncp[l1] = i + 1
            ncp[best_l2] = i + 1

            al1 = float(counts[l1])
            alw = al1 - 1.0
            al2 = float(counts[best_l2])
            alt = al2 + 1.0
            xi = X[i]
            centers[l1] = (centers[l1] * al1 - xi) / alw
            centers[best_l2] = (centers[best_l2] * al2 + xi) / alt
            counts[l1] -= 1
            counts[best_l2] += 1
            an2[l1] = alw / al1
            an1[l1] = big if alw <= 1.0 else alw / (alw - 1.0)
            an1[best_l2] = alt / al2
            an2[best_l2] = alt / (alt + 1.0)
            ic1[i] = best_l2
            ic2[i] = l1

        if indx == n_samples:
            if trace is not None:
                nonzero = np.where(counts > 0)[0]
                mapping = {int(old): int(new) for new, old in enumerate(nonzero)}
                counts_out = counts[nonzero].tolist()
                labels_out = [mapping[int(l)] for l in ic1]
                _emit_trace(
                    trace,
                    "iter_end",
                    start=start,
                    iteration=iteration,
                    moved=False,
                    move_count=0,
                    counts=counts_out,
                    labels=labels_out,
                )
            break

        # QTRAN
        icount = 0
        istep = 0
        while True:
            for i in range(n_samples):
                icount += 1
                istep += 1
                if istep >= i_max_qtr:
                    if trace is not None:
                        nonzero = np.where(counts > 0)[0]
                        mapping = {int(old): int(new) for new, old in enumerate(nonzero)}
                        counts_out = counts[nonzero].tolist()
                        labels_out = [mapping[int(l)] for l in ic1]
                        _emit_trace(
                            trace,
                            "iter_end",
                            start=start,
                            iteration=iteration,
                            moved=bool(indx != n_samples),
                            move_count=int(n_samples - indx),
                            counts=counts_out,
                            labels=labels_out,
                        )
                    return ic1.copy(), centers.copy(), float(np.sum((X - centers[ic1]) ** 2))

                l1 = int(ic1[i])
                l2 = int(ic2[i])

                if counts[l1] == 1:
                    continue

                if istep <= ncp[l1]:
                    d[i] = _squared_distance(X[i], centers[l1]) * an1[l1]

                if not (istep < ncp[l1] or istep < ncp[l2]):
                    continue

                r2 = d[i] / an2[l2]
                dd = 0.0
                for j in range(n_features):
                    de = X[i, j] - centers[l2, j]
                    dd += de * de
                    if dd >= r2:
                        break
                if dd >= r2:
                    continue

                icount = 0
                indx = 0
                itran[l1] = 1
                itran[l2] = 1
                ncp[l1] = istep + n_samples
                ncp[l2] = istep + n_samples

                al1 = float(counts[l1])
                alw = al1 - 1.0
                al2 = float(counts[l2])
                alt = al2 + 1.0
                xi = X[i]
                centers[l1] = (centers[l1] * al1 - xi) / alw
                centers[l2] = (centers[l2] * al2 + xi) / alt
                counts[l1] -= 1
                counts[l2] += 1
                an2[l1] = alw / al1
                an1[l1] = big if alw <= 1.0 else alw / (alw - 1.0)
                an1[l2] = alt / al2
                an2[l2] = alt / (alt + 1.0)
                ic1[i] = l2
                ic2[i] = l1

            if icount == n_samples:
                if trace is not None:
                    nonzero = np.where(counts > 0)[0]
                    mapping = {int(old): int(new) for new, old in enumerate(nonzero)}
                    counts_out = counts[nonzero].tolist()
                    labels_out = [mapping[int(l)] for l in ic1]
                    _emit_trace(
                        trace,
                        "iter_end",
                        start=start,
                        iteration=iteration,
                        moved=bool(indx != n_samples),
                        move_count=int(n_samples - indx),
                        counts=counts_out,
                        labels=labels_out,
                    )
                break

        for cluster_idx in range(n_clusters):
            itran[cluster_idx] = 0
            live[cluster_idx] -= n_samples

        if trace is not None:
            _emit_trace(
                trace,
                "iter_end",
                start=start,
                iteration=iteration,
                moved=bool(indx != n_samples),
                move_count=int(n_samples - indx),
                counts=counts.tolist(),
                labels=ic1.tolist(),
            )

        # R's KMNS ensures there are no empty clusters between iterations
        if np.any(counts == 0):
            ic1, centers, counts = _reseed_empty_clusters(X, ic1, centers, counts)
            # recompute AN1/AN2 and distance heuristic D
            for cluster_idx in range(n_clusters):
                if counts[cluster_idx] > 1:
                    an1[cluster_idx] = counts[cluster_idx] / float(counts[cluster_idx] - 1)
                else:
                    an1[cluster_idx] = big
                an2[cluster_idx] = counts[cluster_idx] / float(counts[cluster_idx] + 1)
            d = d2 * an2[ic2]

        for cluster_idx in range(n_clusters):
            ncp[cluster_idx] = 0

    inertia = float(np.sum((X - centers[ic1]) ** 2))
    return ic1.copy(), centers.copy(), inertia


def hw_kmeans(X, n_clusters, n_start=3, max_iter=50, rng=None, init_idx=None, trace=None):
    """Pure-Python Hartigan-Wong style k-means with multiple random starts.

    This is a straightforward but not highly optimized implementation intended
    to match R's `stats::kmeans` (Hartigan-Wong) update behavior closely.
    It is deterministic when provided a `CentralRNG` or integer seed.

    Parameters
    ----------
    trace : list | callable | None
        Optional trace sink for debugging move order. If provided, each event
        is appended as a dict (or passed to the callback). Events are generic
        and dataset-agnostic, so they can be compared across runs and inputs.
    """
    rng = coerce_rng(random_state=rng, rng=rng)
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    # Determine number of parallel workers from env var (safe default serial)
    try:
        n_jobs = int(os.environ.get("SCDL_NUM_PROCS", "1"))
    except Exception:
        n_jobs = 1

    # Serial single-process path (preserves previous behavior exactly)
    if n_jobs <= 1 or not _parallel_spawn_safe():
        for start in range(max(1, int(n_start))):
            _emit_trace(trace, "start_begin", start=start, n_start=int(n_start), n_samples=n_samples, n_clusters=int(n_clusters), max_iter=int(max_iter))
            # sample initial centers as distinct rows, unless caller provided indices
            if init_idx is not None and start == 0:
                init_idx_arr = np.asarray(init_idx, dtype=int)
            else:
                init_idx_arr = (rng.sample_int(n_samples, n_clusters, replace=False) - 1).astype(int)
            _emit_trace(trace, "start_init", start=start, init_idx=init_idx_arr.tolist())
            labels, centers, inertia = _hartigan_wong_port(X, init_idx_arr, int(max_iter), int(n_clusters), trace=trace, start=start)
            _emit_trace(trace, "start_end", start=start, inertia=float(inertia))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()
        return best_labels, best_centers

    # Parallel multi-start path (parity-safe): precompute init indices deterministically
    ctx = get_context("spawn")
    starts = max(1, int(n_start))
    init_list = []
    for start in range(starts):
        if init_idx is not None and start == 0:
            init_idx_arr = np.asarray(init_idx, dtype=int)
        else:
            init_idx_arr = (rng.sample_int(n_samples, n_clusters, replace=False) - 1).astype(int)
        init_list.append(init_idx_arr)

    # share X via shared_memory to avoid copying to worker processes
    dtype = X.dtype
    shape = X.shape
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    try:
        X_shm = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        X_shm[:] = X[:]

        args = [(shm.name, shape, str(dtype), init_list[i], int(max_iter), int(n_clusters)) for i in range(starts)]
        with ctx.Pool(processes=min(n_jobs, starts)) as pool:
            results = pool.map(_parallel_start_worker, args)

    finally:
        shm.close()
        shm.unlink()

    # choose best start by inertia
    for centers, inertia in results:
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()

    # compute labels for best centers in main process
    best_labels = np.argmin(np.sum((X[:, None, :] - best_centers[None, :, :])**2, axis=2), axis=1)
    return best_labels, best_centers


__all__ = ["hw_kmeans"]
