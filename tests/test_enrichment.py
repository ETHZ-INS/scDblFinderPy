"""Numerical reference from scDblFinder 1.20.2 / MASS::theta.ml."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy.stats import poisson

from scDblFinderPy import doublet_pairwise_enrichment


@pytest.fixture
def stats():
    return pd.DataFrame({
        "combination": list("abcdef"),
        "observed": [5, 12, 25, 7, 30, 19],
        "expected": [10, 8, 15, 12, 20, 10],
        "difficulty": [.2, .3, .4, .5, .6, .7],
    })


def test_r_1202_reference(stats):
    before = stats.copy(deep=True)
    result = doublet_pairwise_enrichment(stats).set_index("combination").loc[list("abcdef")]
    np.testing.assert_allclose(
        result.log2enrich,
        [-1.1565964, .3242854, .5356986, -.8231173, .4687727, .8004082],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result["p.value"], [1, .153845997, .024265181, 1, .029929655, .005698491],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.FDR, [1, .46153799, .12132590, 1, .12132590, .03419095],
        atol=1e-8,
    )
    np.testing.assert_allclose(result.theta, 4.86852695630257, rtol=1e-10)
    pd.testing.assert_frame_equal(stats, before)


def test_anndata_and_h5ad_roundtrip(stats, tmp_path):
    data = AnnData(np.zeros((2, 2)))
    data.uns["scDblFinder_origin_stats"] = stats
    data.write_h5ad(tmp_path / "stats.h5ad")
    restored = read_h5ad(tmp_path / "stats.h5ad")
    pd.testing.assert_frame_equal(
        doublet_pairwise_enrichment(restored), doublet_pairwise_enrichment(stats)
    )


def test_uncorrected_tail_and_holm():
    stats = pd.DataFrame({
        "combination": ["A+B", "A+C", "B+C"],
        "observed": [0, 10, 12], "expected": [2, 2, 2],
    })
    result = doublet_pairwise_enrichment(stats, incl_diff=False).set_index("combination")
    assert result.loc["A+B", "p.value"] == 1
    assert result.loc["A+C", "p.value"] == pytest.approx(poisson.sf(10, 2))
    assert result.loc["B+C", "FDR"] == pytest.approx(3 * poisson.sf(12, 2))
    assert result.loc["A+C", "FDR"] == pytest.approx(2 * poisson.sf(10, 2))
    assert "theta" not in result


@pytest.mark.parametrize("column,value", [
    ("observed", -1), ("observed", .5), ("expected", 0),
    ("difficulty", np.nan), ("difficulty", 0), ("difficulty", 1.1),
])
def test_invalid_statistics_raise(stats, column, value):
    stats[column] = stats[column].astype(float)
    stats.loc[0, column] = value
    with pytest.raises(ValueError):
        doublet_pairwise_enrichment(stats)


def test_missing_duplicate_and_degenerate_statistics(stats):
    with pytest.raises(ValueError, match="Missing"):
        doublet_pairwise_enrichment(stats.drop(columns="difficulty"))
    with pytest.raises(ValueError, match="unique"):
        doublet_pairwise_enrichment(pd.concat([stats, stats]))
    with pytest.raises(ValueError, match="Missing native"):
        doublet_pairwise_enrichment(AnnData(np.zeros((2, 2))))
    with pytest.raises(TypeError):
        doublet_pairwise_enrichment([])
    stats["observed"] = stats.expected
    with pytest.raises(ValueError, match="Dispersion is undefined"):
        doublet_pairwise_enrichment(stats)
    stats["observed"] = 0
    with pytest.raises(ValueError, match="positive total"):
        doublet_pairwise_enrichment(stats)
