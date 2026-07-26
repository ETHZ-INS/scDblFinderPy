import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm as _norm

from .doublet_generation import get_expected_doublets

def _estimate_heterotypic_db_rate(d, dbr=None, dbr_per1k=0.008):
    """Estimate the rate of heterotypic doublets."""
    dbr_global = _gdbr(d, dbr=dbr, dbr_per1k=dbr_per1k)
    
    if "cluster" not in d.columns or d["cluster"].nunique(dropna=True) <= 1:
        th = float(np.sum(d["src"] == "artificial")) / len(d)
        is_art = (d["src"] == "artificial")
        if np.sum(is_art) == 0:
            return dbr_global
        prop_homo = float(np.sum(is_art & (d["score"] < th))) / np.sum(is_art)
        return dbr_global * (1.0 - prop_homo)
        
    d_real = d[d["src"] == "real"]
    if d_real.empty:
        return dbr_global
        
    expected_dict = get_expected_doublets(d_real["cluster"], dbr=dbr_global, only_heterotypic=True, dbr_per_1k=dbr_per1k)
    return float(np.sum(list(expected_dict.values())) / len(d_real))

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
    # Now overwrite dbr_global with the heterotypic rate
    dbr_global = _estimate_heterotypic_db_rate(d, dbr=dbr_global, dbr_per1k=dbr_per1k)
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


def doublet_thresholding_dbr(data, dbr=None, dbr_per1k=0.008):
    """
    Simple quantile-based threshold: score at quantile (1 - heterotypic_dbr).

    Mirrors R's method="dbr" in doubletThresholding. Operates only on real cells
    (src=="real") if the column is present, matching R's behaviour.
    Returns (threshold, calls_array).
    """
    d = data.copy()
    if "src" in d.columns:
        d = d[d["src"] == "real"]
    if d.empty:
        return 0.5, np.array(["singlet"] * len(data))

    dbr_eff = _estimate_heterotypic_db_rate(d, dbr=dbr, dbr_per1k=dbr_per1k)
    th = float(np.quantile(d["score"].to_numpy(dtype=float), 1.0 - dbr_eff))
    calls = np.where(np.asarray(data["score"], dtype=float) > th, "doublet", "singlet")
    return th, calls


def doublet_thresholding_griffiths(data, p=0.1):
    """
    Cluster-wise MAD-based thresholding.

    Mirrors R's method="griffiths" in doubletThresholding. Operates only on
    real cells (src=="real") if the column is present. Returns (thresholds_dict,
    calls_array) where thresholds_dict maps sample → threshold.

    The threshold per sample is the upper tail of a normal distribution fitted
    to the score distribution: median + qnorm(1-p) * mad_const, where mad_const
    is the median of positive deviations scaled by 1.4826 (R's mad() constant).
    """
    _MAD_CONSTANT = 1.4826

    d = data.copy().reset_index(drop=True)
    if "src" in d.columns:
        real_mask = d["src"] == "real"
    else:
        real_mask = pd.Series(True, index=d.index)

    d_real = d[real_mask].copy()
    if d_real.empty:
        return {"all": 0.5}, np.array(["singlet"] * len(data))

    if "sample" not in d_real.columns:
        d_real = d_real.copy()
        d_real["sample"] = "all"

    samples = d_real["sample"].unique()
    thresholds = {}
    for s in samples:
        mask_s = d_real["sample"] == s
        scores_s = d_real.loc[mask_s, "score"].to_numpy(dtype=float)
        med = float(np.median(scores_s))
        devs = scores_s - med
        pos_devs = devs[devs > 0]
        mad_est = (float(np.median(pos_devs)) * _MAD_CONSTANT) if pos_devs.size > 0 else 1e-6
        # qnorm(p, mean=med, sd=mad_est, lower.tail=FALSE) = med + qnorm(1-p)*mad_est
        th_s = med + float(_norm.ppf(1.0 - p)) * mad_est
        thresholds[s] = th_s

    # Map threshold back to all rows in data
    scores_all = np.asarray(data["score"], dtype=float)
    if "sample" in data.columns:
        row_th = data["sample"].map(lambda s: thresholds.get(s, thresholds.get("all", 0.5)))
    else:
        row_th = pd.Series([thresholds.get("all", list(thresholds.values())[0])] * len(data))
    calls = np.where(scores_all > row_th.to_numpy(), "doublet", "singlet")
    return thresholds, calls


def doublet_thresholding(data, method="auto", dbr=None, dbr_sd=None,
                          stringency=0.5, p=0.1, dbr_per1k=0.008,
                          return_type="call"):
    """
    Dispatcher for doublet thresholding. Mirrors R's doubletThresholding().

    Parameters
    ----------
    data : pd.DataFrame
        Must contain at minimum a 'score' column. 'type' and 'src' enable
        richer methods.
    method : str
        'auto' (default), 'optim', 'dbr', or 'griffiths'.
        'auto' selects 'optim' when 'type' is present, otherwise 'dbr'.
    dbr : float or None
        Expected doublet rate. Auto-estimated when None.
    dbr_sd : float or None
        Uncertainty in doublet rate (optim only). Defaults to 0.4*dbr.
    stringency : float
        Weight of false positives vs false negatives (optim only, 0 < s < 1).
    p : float
        P-value threshold for griffiths method.
    dbr_per1k : float
        Doublet rate per 1000 cells (used when dbr is None).
    return_type : str
        'call' (default) returns singlet/doublet labels.
        'threshold' returns the numeric threshold(s).

    Returns
    -------
    np.ndarray or float or dict
        When return_type='call': array of 'singlet'/'doublet' labels.
        When return_type='threshold': scalar threshold (optim/dbr) or dict
        of per-sample thresholds (griffiths).
    """
    if "score" not in data.columns:
        raise ValueError("`data` must contain a 'score' column.")

    if method == "auto":
        if "type" in data.columns:
            method = "optim"
        else:
            method = "dbr"

    if method == "optim":
        th, calls = doublet_thresholding_optim(
            data, dbr=dbr, dbr_sd=dbr_sd, stringency=stringency, dbr_per1k=dbr_per1k
        )
        return th if return_type == "threshold" else calls

    if method == "dbr":
        th, calls = doublet_thresholding_dbr(data, dbr=dbr, dbr_per1k=dbr_per1k)
        return th if return_type == "threshold" else calls

    if method == "griffiths":
        th_dict, calls = doublet_thresholding_griffiths(data, p=p)
        return th_dict if return_type == "threshold" else calls

    raise ValueError(f"Unknown thresholding method '{method}'. Choose from 'auto', 'optim', 'dbr', 'griffiths'.")
