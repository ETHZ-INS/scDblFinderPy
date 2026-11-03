import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy import sparse

from scDblFinderPy import scDblFinder as engine


@pytest.mark.parametrize("return_type", ["adata", "full"])
@pytest.mark.parametrize("clusters_col", ["celltype", None])
def test_real_cell_origins_match_internal_values(
    monkeypatch, tmp_path, return_type, clusters_col
):
    rng = np.random.default_rng(42)
    data = AnnData(
        sparse.csr_matrix(rng.poisson(3, (80, 50))),
        obs=pd.DataFrame(
            {"celltype": ["Tcell", "Tumor"] * 40},
            index=[f"cell{i}" for i in rng.permutation(80)],
        ),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(50)]),
    )
    before = data.copy()
    expected = {}
    evaluate_knn = engine._evaluate_knn

    def capture_origins(combined, **kwargs):
        features = evaluate_knn(combined, **kwargs)
        real = combined.obs["type"].eq("real").to_numpy()
        real_indices = np.flatnonzero(real)
        # Exercise missing and ambiguous origins in the exported annotations.
        features["most_likely_origin"][real_indices[0]] = np.nan
        features["origin_ambiguous"][real_indices[1]] = True
        ids = combined.obs.loc[real, "_orig_index"].to_numpy()
        for source, target in (
            ("most_likely_origin", "scDblFinder_mostLikelyOrigin"),
            ("origin_ambiguous", "scDblFinder_originAmbiguous"),
        ):
            expected[target] = pd.Series(features[source][real], index=ids)
        return features

    monkeypatch.setattr(engine, "_evaluate_knn", capture_origins)
    result = engine.compute_doublet_score(
        data,
        clusters_col=clusters_col,
        random_state=123,
        n_features=30,
        n_components=10,
        n_artificial=100,
        n_neighbors=10,
        n_iters=1,
        nrounds=10,
        filter_unidentifiable=False,
        return_type=return_type,
        verbose=False,
    )

    real_obs = result.obs.iloc[:data.n_obs]
    assert real_obs.index.equals(data.obs_names)
    for column, values in expected.items():
        pd.testing.assert_series_equal(
            real_obs[column], values.reindex(data.obs_names),
            check_names=False, check_dtype=False, check_categorical=False,
        )
    assert pd.isna(real_obs["scDblFinder_mostLikelyOrigin"].iloc[0])
    assert bool(real_obs["scDblFinder_originAmbiguous"].iloc[1])
    assert real_obs["scDblFinder_score"].notna().all()
    assert real_obs["scDblFinder_class"].isin(["singlet", "doublet"]).all()
    assert (result[before.obs_names, before.var_names].X != before.X).nnz == 0
    pd.testing.assert_frame_equal(data.obs[before.obs.columns], before.obs)
    if return_type == "full":
        assert result.n_obs > data.n_obs
        assert "most_likely_origin" in result.obs
    else:
        assert result.shape == data.shape
    if return_type == "adata":
        path = tmp_path / "origins.h5ad"
        result.write_h5ad(path)
        restored = read_h5ad(path)
        for column in expected:
            pd.testing.assert_series_equal(
                restored.obs[column], real_obs[column],
                check_dtype=False, check_categorical=False,
            )
