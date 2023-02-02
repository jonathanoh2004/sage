import numpy as np
import sympy
from tedana.workflows.sage import loglinear_sage


def test__get_ind_vars():
    tes = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    res_exp = np.array(
        [[1, 0, -1, 0], [1, 0, -2, 0], [1, -1, -2, -1], [1, -1, -1, -3], [1, -1, 0, -5]],
        dtype=np.float64,
    )
    res_test = loglinear_sage._get_ind_vars(tes)

    np.testing.assert_array_equal(res_test, res_exp)


def _get_test_data(shape):
    d1 = np.arange(4).reshape((shape[0], 1, 1))
    d2 = (np.arange(3) + 1).reshape((1, shape[1], 1))
    d3 = ((np.arange(2) + 1) * 7).reshape((1, 1, shape[2]))
    return d1 * d2 * d3


def _get_test_mask(shape):
    mask = np.array([1, 1, 0, 1]).reshape((shape[0], 1, 1))
    return mask


def test__get_dep_vars():
    shape = (4, 3, 2)
    data = _get_test_data(shape)
    mask = _get_test_mask(shape)

    res_exp = np.array(
        [
            [
                [
                    0.0,
                    0.0,
                    1.945910149055,
                    2.639057329615,
                    0.0,
                    0.0,
                    3.044522437723,
                    3.737669618283,
                ],
                [
                    0.0,
                    0.0,
                    2.639057329615,
                    3.332204510175,
                    0.0,
                    0.0,
                    3.737669618283,
                    4.430816798843,
                ],
                [
                    0.0,
                    0.0,
                    3.044522437723,
                    3.737669618283,
                    0.0,
                    0.0,
                    4.143134726392,
                    4.836281906951,
                ],
            ]
        ]
    )

    res_test = loglinear_sage._get_dep_vars(data, mask)

    np.testing.assert_allclose(res_test, res_exp, rtol=1e-12)


def test_loglinear():
    # s0I, delta, r2star, r2 = sympy.symbols("s0I delta r2* r2")
    data = np.ones((2, 5, 2))
    data[:] = data[:] * np.exp(((np.arange(5) * 2)[::-1] + 1)[np.newaxis, :, np.newaxis])
    mask = np.array([1, 0]).astype(bool)
    tes = np.array([1, 2, 3, 4, 5]).astype(np.float64)

    # s0I_exp, delta_exp, r2star_exp, r2_exp = sympy.symbols("s_0I delta r2* r2")
    A = sympy.Matrix(
        [[1, 0, -1, 0], [1, 0, -2, 0], [1, -1, -2, -1], [1, -1, -1, -3], [1, -1, 0, -5]]
    )
    B = sympy.Matrix(
        [
            [9.0, 9.0, 0.0, 0.0],
            [7.0, 7.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    res_exp = A.pinv_solve(B)
    s0I_exp = np.exp(np.array(res_exp[0, :]).astype(np.float64)).T.reshape(2, 2)
    delta_exp = np.exp(np.array(res_exp[1, :]).astype(np.float64)).T.reshape(2, 2)
    s0II_exp = s0I_exp / delta_exp
    t2star_exp = 1 / np.array(res_exp[2, :]).astype(np.float64).T.reshape(2, 2)
    t2_exp = 1 / np.array(res_exp[3, :]).astype(np.float64).T.reshape(2, 2)

    (
        t2star_test,
        s0I_test,
        t2_test,
        s0II_test,
        delta_test,
        rmspe_test,
    ) = loglinear_sage.get_maps_loglinear(data, tes, mask, None)

    assert rmspe_test == None
    np.testing.assert_allclose(s0I_test, s0I_exp, rtol=1e-12)
    np.testing.assert_allclose(delta_test, delta_exp, rtol=1e-12)
    np.testing.assert_allclose(s0II_test, s0II_exp, rtol=1e-12)
    np.testing.assert_allclose(t2star_test, t2star_exp, rtol=1e-12)
    np.testing.assert_allclose(t2_test, t2_exp, rtol=1e-12)
