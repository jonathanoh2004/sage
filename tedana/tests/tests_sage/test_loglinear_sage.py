import os.path

import numpy as np
import pytest
import nilearn

from tedana import io, utils
from tedana.tests.utils_sage import get_test_data_path_sage
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
    d1 = np.arange(4).reshape(shape[0], 1, 1)
    d2 = (np.arange(3) + 1).reshape(1, shape[1], 1)
    d3 = ((np.arange(2) + 1) * 7).reshape(1, 1, shape[2])


def _get_test_mask(shape):
    mask = np.array([1, 1, 0, 1]).reshape(shape[0], 1, 1)


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
