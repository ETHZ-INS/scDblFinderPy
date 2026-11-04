"""Pairwise enrichment following R/Bioconductor scDblFinder 1.20.2.

Reference: Pierre-Luc Germain et al., scDblFinder (GPL-3),
https://github.com/plger/scDblFinder, doubletPairwiseEnrichment.
Dispersion estimation follows MASS::theta.ml; the offset GLM follows
MASS::negative.binomial and stats::glm.fit initialization/convergence.
"""

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.special import digamma, polygamma, xlogy
from scipy.stats import poisson
from statsmodels.stats.multitest import multipletests


def _estimate_theta(y, mu):
    discrepancy = np.sum((y / mu - 1) ** 2)
    if discrepancy == 0:
        raise ValueError("Dispersion is undefined when observed equals expected for every pair.")
    theta = len(y) / discrepancy
    delta = 1.0
    for _ in range(99):
        if abs(delta) <= np.finfo(float).eps ** 0.25:
            break
        theta = abs(theta)
        if not np.isfinite(theta) or theta <= 0:
            raise RuntimeError("Invalid negative-binomial dispersion estimate.")
        score = np.sum(
            digamma(theta + y) - digamma(theta) + np.log(theta)
            + 1 - np.log(theta + mu) - (y + theta) / (mu + theta)
        )
        information = np.sum(
            -polygamma(1, theta + y) + polygamma(1, theta) - 1 / theta
            + 2 / (mu + theta) - (y + theta) / (mu + theta) ** 2
        )
        if not np.isfinite(information) or information == 0:
            raise RuntimeError("Dispersion estimation failed; no fallback was applied.")
        delta = score / information
        theta += delta
    if not np.isfinite(theta) or theta <= 0:
        raise RuntimeError("Invalid negative-binomial dispersion estimate.")
    if abs(delta) > np.finfo(float).eps ** 0.25:
        warnings.warn("Dispersion estimation reached its iteration limit.", RuntimeWarning)
    return theta


def _adjust_expectation(y, mu, difficulty):
    theta = _estimate_theta(y, mu)
    predictor = np.log(difficulty)
    offset = np.log(mu)
    fitted = y + (y == 0) / 6
    eta = np.log(fitted)

    def deviance(mean):
        return 2 * np.sum(
            xlogy(y, y / mean)
            - (y + theta) * np.log((y + theta) / (mean + theta))
        )

    previous = deviance(fitted)
    for _ in range(25):
        weights = fitted / (1 + fitted / theta)
        working = eta - offset + (y - fitted) / fitted
        denominator = np.sum(weights * predictor ** 2)
        if denominator <= 0 or not np.isfinite(denominator):
            raise RuntimeError("Difficulty correction has a singular design.")
        coefficient = np.sum(weights * predictor * working) / denominator
        eta = offset + predictor * coefficient
        fitted = np.exp(eta)
        if not np.isfinite(fitted).all() or (fitted <= 0).any():
            raise RuntimeError("Difficulty correction produced invalid expectations.")
        current = deviance(fitted)
        if abs(current - previous) / (0.1 + abs(current)) < 1e-8:
            return fitted, theta, coefficient
        previous = current
    raise RuntimeError("Difficulty correction GLM did not converge.")


def doublet_pairwise_enrichment(data, *, incl_diff=True):
    """Test enrichment of heterotypic doublet origins for one capture/run.

    Parameters
    ----------
    data : AnnData or pandas.DataFrame
        Cluster-based detector output with ``uns['scDblFinder_origin_stats']``,
        or a statistics table containing unique ``combination`` names,
        integer ``observed`` counts and positive ``expected`` counts.
        Default correction also requires finite ``difficulty`` in (0, 1].
        Per-cell Origin labels or consensus calls alone are insufficient.
    incl_diff : bool, default True
        Correct expectations with a negative-binomial log-link GLM:
        no intercept, log(expected) offset, log(difficulty) predictor.
        False explicitly tests unadjusted expected counts.

    Returns
    -------
    pandas.DataFrame
        Input pair statistics plus ``expected_adjusted``, ``log2enrich``,
        ``p.value`` and ``FDR``, sorted by p-value then absolute effect.
        FDR follows R's column name but is a Holm-adjusted p-value (FWER).
        Corrected output also includes dispersion and GLM coefficient.

    Notes
    -----
    Implements the R 1.20.2 default upper-tail Poisson test, including its
    strict P(X > observed) tail and log2((observed+1)/(expected+1)) effect.
    Depleted pairs receive p=1. It does not implement R's other test types,
    lower-tail or sample pooling options. Call separately per capture/seed.
    Missing/zero difficulties raise rather than being imputed or silently
    dropped; explicitly selecting a valid subset changes the test family.
    Neither predictions nor enrichment establish physical interaction.
    The input is not modified and no random numbers are used.
    """
    if not isinstance(incl_diff, (bool, np.bool_)):
        raise TypeError("incl_diff must be boolean.")
    if isinstance(data, AnnData):
        if "scDblFinder_origin_stats" not in data.uns:
            raise ValueError("Missing native Origin statistics; rerun clustered detection.")
        data = data.uns["scDblFinder_origin_stats"]
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be AnnData or a pandas DataFrame.")
    required = ["combination", "observed", "expected"]
    if incl_diff:
        required.append("difficulty")
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Missing native Origin statistics: {sorted(missing)}")
    result = data.copy(deep=True)
    if (result.empty or result.combination.isna().any()
            or result.combination.duplicated().any()):
        raise ValueError("Each combination must be present and unique for one capture/run.")
    values = result[required[1:]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:, 0] < 0).any()
            or not np.equal(values[:, 0], np.floor(values[:, 0])).all()
            or (values[:, 1] <= 0).any()):
        raise ValueError("Require finite integer observations >=0 and expectations >0.")
    y, mu = values[:, 0], values[:, 1]
    if incl_diff:
        if (len(result) < 2 or y.sum() == 0
                or (values[:, 2] <= 0).any() or (values[:, 2] > 1).any()):
            raise ValueError("Correction requires >=2 pairs, positive total and difficulty in (0, 1].")
        adjusted, theta, coefficient = _adjust_expectation(y, mu, values[:, 2])
        result["theta"] = theta
        result["difficulty_coefficient"] = coefficient
    else:
        adjusted = mu.copy()
    effect = np.log2((1 + y) / (1 + adjusted))
    pvalues = poisson.sf(y, adjusted)
    pvalues[effect < 0] = 1
    result["expected_adjusted"] = adjusted
    result["log2enrich"] = effect
    result["p.value"] = pvalues
    result["FDR"] = multipletests(pvalues, method="holm")[1]
    order = np.lexsort((-np.abs(effect), pvalues))
    return result.iloc[order].reset_index(drop=True)
