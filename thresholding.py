import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _prop_homotypic(clusters):
    """Estimate expected homotypic-pair proportion from cluster frequencies."""
    clusters = np.asarray(clusters)
    if clusters.size == 0:
        return 0.0
    _, counts = np.unique(clusters, return_counts=True)
    p = counts / counts.sum()
    return float(np.sum(p * p))


def _gdbr(d, dbr=None, dbr_per1k=0.008):
    """R-like global doublet rate helper (.gdbr)."""
    if dbr is not None:
        if np.isscalar(dbr):
            return float(dbr)
        if "sample" not in d.columns:
            raise ValueError("If `dbr` is per-sample, `sample` must be present in the data.")
        rates = pd.Series(dbr)
        real_counts = d.loc[d["src"] == "real", "sample"].value_counts()
        matched = rates.reindex(real_counts.index)
        if matched.isna().any():
            raise ValueError("Per-sample `dbr` names do not match sample labels in the data.")
        return float(np.sum(matched.values * real_counts.values) / np.sum(real_counts.values))

    if "sample" not in d.columns:
        sl = np.array([int((d["src"] == "real").sum())], dtype=float)
    else:
        sl = d.loc[d["src"] == "real", "sample"].value_counts().values.astype(float)

    sample_rates = dbr_per1k * sl / 1000.0
    return float(np.sum(sample_rates * sl) / np.sum(sl))


def _fpr(type_is_real, score, threshold):
    type_is_real = np.asarray(type_is_real, dtype=bool)
    score = np.asarray(score, dtype=float)
    if type_is_real.size == 0:
        return 0.0
    denom = np.sum(type_is_real)
    if denom == 0:
        return 0.0
    return float(np.sum(type_is_real & (score >= threshold)) / denom)


def _fnr(type_is_real, score, threshold, expected_fn=0.0):
    type_is_real = np.asarray(type_is_real, dtype=bool)
    score = np.asarray(score, dtype=float)
    n_doublet = np.sum(~type_is_real)
    if n_doublet == 0:
        return 0.0
    observed_fn = np.sum((~type_is_real) & (score < threshold))
    return float(max(0.0, observed_fn - expected_fn) / n_doublet)


def _prop_dev(type_is_real, score, expected, threshold):
    type_is_real = np.asarray(type_is_real, dtype=bool)
    score = np.asarray(score, dtype=float)

    x = 1.0 + np.sum((score >= threshold) & type_is_real)
    expected = np.asarray(expected, dtype=float) + 1.0

    if expected.size > 1 and (x > np.min(expected)) and (x < np.max(expected)):
        return 0.0
    return float(np.min(np.abs(x - expected) / expected))


def optim_threshold(data, dbr=None, dbr_sd=None, stringency=0.5, dbr_per1k=0.008):
    """
    Optimize threshold using the R `.optimThreshold` cost structure.

    Required columns: `type`, `src`, `score`.
    Optional columns: `cluster`, `include.in.training`, `sample`.
    """
    if not (0.0 < stringency < 1.0):
        raise ValueError("`stringency` should be >0 and <1.")

    required = {"type", "src", "score"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns for thresholding: {sorted(missing)}")

    d = data.copy()

    if "cluster" not in d.columns:
        d["cluster"] = 1
    if "include.in.training" not in d.columns:
        d["include.in.training"] = True

    d["type_is_real"] = d["type"].astype(str) == "real"

    dbr_global = _gdbr(d, dbr=dbr, dbr_per1k=dbr_per1k)
    if dbr_sd is None:
        dbr_sd = 0.4 * dbr_global
    dbr_bounds = np.array([dbr_global], dtype=float)
    if dbr_sd is not None:
        dbr_bounds = np.array([max(0.0, dbr_global - dbr_sd), min(1.0, dbr_global + dbr_sd)], dtype=float)

    n_real = int((d["src"] == "real").sum())
    expected = dbr_bounds * n_real

    # R eFN uses rowname pattern '^rDbl\.'; this naming is absent in current Python flow.
    efn = 0.0
    real_clusters = d.loc[d["src"] == "real", "cluster"]
    if real_clusters.nunique(dropna=True) > 1:
        # Keep parity with R structure while defaulting to 0 when no recoverable rDbl rows exist.
        row_labels = d.index.astype(str)
        n_rdbl = np.sum(np.char.startswith(row_labels.values.astype(str), "rDbl."))
        if n_rdbl > 0:
            efn = float(n_rdbl * _prop_homotypic(real_clusters.dropna().values))

    include_mask = d["include.in.training"].to_numpy(dtype=bool)
    scores = d["score"].to_numpy(dtype=float)
    type_is_real = d["type_is_real"].to_numpy(dtype=bool)

    def cost_fn(th):
        dev = _prop_dev(type_is_real, scores, expected, th) ** 2
        val = dev + 2.0 * (1.0 - stringency) * _fnr(type_is_real, scores, th, expected_fn=efn)
        if include_mask.size > 0:
            val += 2.0 * stringency * _fpr(type_is_real[include_mask], scores[include_mask], th)
        return val

    res = minimize_scalar(cost_fn, bounds=(0.0, 1.0), method="bounded")
    if not res.success:
        return 0.5
    return float(res.x)


def doublet_thresholding_optim(data, dbr=None, dbr_sd=None, stringency=0.5, dbr_per1k=0.008):
    """Return both threshold and singlet/doublet calls for the optim method."""
    th = optim_threshold(
        data,
        dbr=dbr,
        dbr_sd=dbr_sd,
        stringency=stringency,
        dbr_per1k=dbr_per1k,
    )
    calls = np.where(np.asarray(data["score"], dtype=float) > th, "doublet", "singlet")
    return th, calls
