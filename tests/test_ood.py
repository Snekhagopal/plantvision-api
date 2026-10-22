import numpy as np

from src.inference.ood import MahalanobisScorer


def naive_score(scorer, f):
    d = []
    for x in np.atleast_2d(f):
        diffs = x[None, :] - scorer.class_means
        d.append(np.sqrt(min(float(v @ scorer.precision @ v) for v in diffs)))
    return np.array(d)


def make_data(seed=0, n=300, d=16, c=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, c, n)
    centers = rng.normal(0, 5, (c, d))
    return centers[labels] + rng.normal(0, 1, (n, d)), labels


def test_vectorised_score_matches_naive():
    x, y = make_data()
    scorer = MahalanobisScorer.fit(x, y, 3)
    q = np.random.default_rng(1).normal(0, 3, (20, 16))
    assert np.allclose(scorer.score(q), naive_score(scorer, q), rtol=1e-6, atol=1e-6)


def test_far_points_score_higher_than_training_points():
    x, y = make_data()
    scorer = MahalanobisScorer.fit(x, y, 3)
    in_dist = scorer.score(x).mean()
    far = scorer.score(x + 50.0).mean()
    assert far > 5 * in_dist


def test_save_load_roundtrip(tmp_path):
    x, y = make_data()
    scorer = MahalanobisScorer.fit(x, y, 3)
    scorer.save(tmp_path / "s.npz")
    loaded = MahalanobisScorer.load(tmp_path / "s.npz")
    assert np.allclose(loaded.score(x[:5]), scorer.score(x[:5]), rtol=1e-4)
