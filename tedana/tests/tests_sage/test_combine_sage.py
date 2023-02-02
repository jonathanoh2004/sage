import numpy as np
import sympy
from unittest.mock import patch

from tedana.workflows.sage import combine_sage

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4254163/


@patch("tedana.workflows.sage.combine_sage.weights_sage", spec_set=combine_sage.weights_sage)
def test_make_optcom_sage(mock_weights_sage):
    w_t2star = np.arange(20).reshape(2, 5, 2) * 0.05
    w_t2 = np.arange(20).reshape(2, 5, 2) * 0.075
    mock_weights_sage.side_effect = lambda arg1, arg2, arg3, arg4, arg5: [w_t2star, w_t2]
    data = np.ones((2, 5, 2))
    tes = np.array([5, 4, 3, 2, 1])
    mask = np.array([1, 0])[:, np.newaxis]
    t2star, s0I, t2, s0II = (
        np.ones((2, 5, 2)),
        np.ones((2, 5, 2)),
        np.ones((2, 5, 2)),
        np.ones((2, 5, 2)),
    )

    optcom_t2star_test, optcom_t2_test = combine_sage.make_optcom_sage(
        data, tes, t2star, s0I, t2, s0II, mask
    )

    optcom_t2star_exp = np.array([[1.0, 1.25], [0.0, 0.0]])
    optcom_t2_exp = np.array([[1.5, 1.875], [0.0, 0.0]])
    np.testing.assert_allclose(optcom_t2star_test, optcom_t2star_exp, rtol=1e-12)
    np.testing.assert_allclose(optcom_t2_test, optcom_t2_exp, rtol=1e-12)


def test_weights_sage():
    tes_in = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    r2star_in = np.ones((2, 2)) * 20
    s0I_in = np.ones((2, 2)) * 200
    r2_in = np.ones((2, 2)) * 15
    s0II_in = np.ones((2, 2)) * 150

    idx_I = tes_in < tes_in[-1] / 2
    idx_II = tes_in > tes_in[-1] / 2

    ##### Compute Expected Values #####
    s0I, s0II, delta, r2star, r2, t, tese = sympy.symbols(
        "s_0I s_0II delta r_2^* r_2 tau TE_SE", real=True
    )
    e1 = s0I * sympy.exp(-t * r2star)
    e2 = s0II * sympy.exp(-tese * (r2star - r2)) * sympy.exp(-t * ((2 * r2) - r2star))

    e1_r2star = sympy.diff(e1, r2star)
    e1_r2 = sympy.diff(e1, r2)
    e2_r2star = sympy.diff(e2, r2star)
    e2_r2 = sympy.diff(e2, r2)

    f_e1_r2star = sympy.lambdify((s0I, r2star, t), e1_r2star, "numpy")
    f_e1_r2 = sympy.lambdify((s0I, r2star, t), e1_r2, "numpy")
    f_e2_r2star = sympy.lambdify((s0II, r2star, r2, t, tese), e2_r2star, "numpy")
    f_e2_r2 = sympy.lambdify((s0II, r2star, r2, t, tese), e2_r2, "numpy")

    w_t2star_1 = f_e1_r2star(
        s0I_in[:, np.newaxis, :],
        r2star_in[:, np.newaxis, :],
        tes_in[np.newaxis, idx_I, np.newaxis],
    )
    w_t2_1 = f_e1_r2(
        s0I_in[:, np.newaxis, :],
        r2star_in[:, np.newaxis, :],
        tes_in[np.newaxis, idx_I, np.newaxis],
    )

    w_t2star_2 = f_e2_r2star(
        s0II_in[:, np.newaxis, :],
        r2star_in[:, np.newaxis, :],
        r2_in[:, np.newaxis, :],
        tes_in[np.newaxis, idx_II, np.newaxis],
        tes_in[-1],
    )
    w_t2_2 = f_e2_r2(
        s0II_in[:, np.newaxis, :],
        r2star_in[:, np.newaxis, :],
        r2_in[:, np.newaxis, :],
        tes_in[np.newaxis, idx_II, np.newaxis],
        tes_in[-1],
    )

    if w_t2_1 == 0:
        w_t2_1 = np.zeros((2, tes_in[idx_I].size, 2))

    w_t2star_exp = np.concatenate([w_t2star_1, w_t2star_2], axis=1)
    w_t2_exp = np.concatenate([w_t2_1, w_t2_2], axis=1)

    w_t2star_exp = w_t2star_exp / np.expand_dims(np.sum(w_t2star_exp, axis=1), axis=1)
    w_t2_exp = w_t2_exp / np.expand_dims(np.sum(w_t2_exp, axis=1), axis=1)

    ##### Compute Test Values #####
    w_t2star_test, w_t2_test = combine_sage.weights_sage(
        tes_in, 1 / r2star_in, s0I_in, 1 / r2_in, s0II_in
    )

    np.testing.assert_allclose(w_t2star_test, w_t2star_exp, rtol=1e-12)
    np.testing.assert_allclose(w_t2_test, w_t2_exp, rtol=1e-12)
