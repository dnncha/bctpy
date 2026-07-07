import numpy as np
import bct

# seeds and parameters
N = 20
DENSITY = 0.5
BIN_SWAPS = 5
WEI_FREQ = 0.1
N_SEEDS = 20

def _make_symmetric_signed(n, density, seed):
    # fixed-seed symmetric matrix with mixed positive/negative weights
    rng = np.random.RandomState(seed)
    W = rng.randn(n, n)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    mask = rng.random_sample((n, n)) < density
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    W *= mask
    return W


def _make_directed_signed(n, density, seed):
    # fixed-seed directed matrix with genuine asymmetry
    rng = np.random.RandomState(seed)
    W = rng.randn(n, n)
    np.fill_diagonal(W, 0)
    mask = rng.random_sample((n, n)) < density
    np.fill_diagonal(mask, False)
    W *= mask
    # force asymmetry: opposing signs across the diagonal
    for i in range(min(5, n)):
        for j in range(i + 1, min(i + 3, n)):
            W[i, j] = rng.uniform(0.1, 1.0)
            W[j, i] = rng.uniform(-1.0, -0.1)
    return W


# sign preservation, undirected

def test_null_model_und_sign_has_negative_edges():
    # output must contain negative edges when input does
    x = _make_symmetric_signed(N, DENSITY, seed=42)
    neg_in = np.sum(x < 0)
    print('input neg edges', neg_in)
    assert neg_in > 0

    W0, _ = bct.null_model_und_sign(x, BIN_SWAPS, WEI_FREQ, seed=7)
    neg_out = np.sum(W0 < 0)
    print('output neg edges', neg_out)
    assert neg_out > 0

# sign preservation, directed

def test_null_model_dir_sign_has_negative_edges():
    # output must contain negative edges when input does
    x = _make_directed_signed(N, DENSITY, seed=42)
    neg_in = np.sum(x < 0)
    print('input neg edges', neg_in)
    assert neg_in > 0

    W0, _ = bct.null_model_dir_sign(x, BIN_SWAPS, WEI_FREQ, seed=7)
    neg_out = np.sum(W0 < 0)
    print('output neg edges', neg_out)
    assert neg_out > 0

# degree distribution preservation (regression)

def test_null_model_und_sign_degree_distribution():
    # positive and negative degree distributions must be preserved exactly
    x = _make_symmetric_signed(N, DENSITY, seed=99)
    pos_deg_in = np.sort(np.sum(x > 0, axis=0))
    neg_deg_in = np.sort(np.sum(x < 0, axis=0))

    for s in range(N_SEEDS):
        W0, _ = bct.null_model_und_sign(x, BIN_SWAPS, WEI_FREQ, seed=s)
        pos_deg_out = np.sort(np.sum(W0 > 0, axis=0))
        neg_deg_out = np.sort(np.sum(W0 < 0, axis=0))
        print('seed', s, 'pos_deg match', np.array_equal(pos_deg_out, pos_deg_in),
              'neg_deg match', np.array_equal(neg_deg_out, neg_deg_in))
        assert np.array_equal(pos_deg_out, pos_deg_in)
        assert np.array_equal(neg_deg_out, neg_deg_in)


def test_null_model_dir_sign_degree_distribution():
    # positive and negative in/out-degree distributions must be preserved
    x = _make_directed_signed(N, DENSITY, seed=99)
    pos_out_in = np.sort(np.sum(x > 0, axis=1))
    pos_in_in = np.sort(np.sum(x > 0, axis=0))
    neg_out_in = np.sort(np.sum(x < 0, axis=1))
    neg_in_in = np.sort(np.sum(x < 0, axis=0))

    for s in range(N_SEEDS):
        W0, _ = bct.null_model_dir_sign(x, BIN_SWAPS, WEI_FREQ, seed=s)
        print('seed', s)
        assert np.array_equal(np.sort(np.sum(W0 > 0, axis=1)), pos_out_in)
        assert np.array_equal(np.sort(np.sum(W0 > 0, axis=0)), pos_in_in)
        assert np.array_equal(np.sort(np.sum(W0 < 0, axis=1)), neg_out_in)
        assert np.array_equal(np.sort(np.sum(W0 < 0, axis=0)), neg_in_in)
