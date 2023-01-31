import os.path
from unittest.mock import patch, call, MagicMock
import numpy as np
import pytest
import nilearn
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

from tedana import io, utils
from tedana.tests.utils_sage import get_test_data_path_sage
from tedana.workflows.sage import nonlinear_sage, nonlinear_3param_sage, nonlinear_4param_sage


def _mock_curve_fit(model, X, Y, p0, bounds, ftol, xtol, max_nfev):
    return [0, 1, 2, 3], None


def _mock_prep_shared_mem_with_name(shr_mem, shape, dtype):
    if "Y" in shr_mem:
        return {"Y": None}, {"Y": np.zeros(shape)}
    elif "X" in shr_mem:
        return {"X": None}, {"X": np.zeros(shape)}
    else:
        return {
            "r2star_guess": None,
            "s0I_guess": None,
            "r2_guess": None,
            "s0II_guess": None,
            "delta_res": None,
            "r2star_res": None,
            "s0I_res": None,
            "r2_res": None,
            "s0II_res": None,
            "rmspe_res": None,
        }, {
            "r2star_guess": np.zeros(shape),
            "s0I_guess": np.zeros(shape),
            "r2_guess": np.zeros(shape),
            "s0II_guess": np.zeros(shape),
            "delta_res": np.zeros(shape),
            "r2star_res": np.zeros(shape),
            "s0I_res": np.zeros(shape),
            "r2_res": np.zeros(shape),
            "s0II_res": np.zeros(shape),
            "rmspe_res": np.zeros(shape),
        }


@patch(
    "tedana.workflows.sage.nonlinear_sage.concurrency_sage.prep_shared_mem_with_name",
    spec_set=_mock_prep_shared_mem_with_name,
)
@patch(
    "tedana.workflows.sage.nonlinear_sage.concurrency_sage.close_shr_mem",
    spec_set=(lambda: {"": None}),
)
@patch("tedana.workflows.sage.nonlinear_sage.curve_fit", spec_set=_mock_curve_fit)
def test_nonlinear_4param_sage(mock1, mock2, mock3):
    mock1.side_effect = _mock_curve_fit
    mock3.side_effect = _mock_prep_shared_mem_with_name
    nonlin_fitter = nonlinear_4param_sage.Get_Maps_Nonlinear_4Param(n_param=4)
    nonlin_fitter.get_bounds = MagicMock()
    nonlin_fitter.get_max_iter = MagicMock()
    nonlin_fitter.get_guesses = MagicMock()
    nonlin_fitter.get_model = MagicMock(return_value=lambda: None)
    nonlin_fitter.eval_model = MagicMock(return_value=0)

    shape = (3, 3, 3)
    dtype = np.float64
    t_start = 0
    t_end = 3
    Y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    r2star_guess = ""
    s0I_guess = ""
    r2_guess = ""
    s0II_guess = ""
    delta_res = ""
    r2star_res = ""
    s0I_res = ""
    r2_res = ""
    s0II_res = ""
    rmspe_res = ""

    nonlin_fitter.fit_nonlinear_sage(
        shape,
        dtype,
        t_start,
        t_end,
        Y,
        X,
        r2star_guess,
        s0I_guess,
        r2_guess,
        s0II_guess,
        delta_res,
        r2star_res,
        s0I_res,
        r2_res,
        s0II_res,
        rmspe_res,
    )

    num_calls_exp = 3
    num_calls_curve_fit_exp = 9

    assert mock3.call_count == num_calls_exp
    assert mock2.call_count == num_calls_exp
    assert mock1.call_count == num_calls_curve_fit_exp

    assert nonlin_fitter.get_bounds.call_count == 1
    assert nonlin_fitter.get_max_iter.call_count == 1
    assert nonlin_fitter.get_guesses.call_count == 9
    assert nonlin_fitter.get_model.call_count == 9
    assert nonlin_fitter.eval_model.call_count == 9


"""
shape
dtype
t_start
t_end
Y
X
r2star_guess
s0I_guess
r2_guess
s0II_guess
delta_res
r2star_res
s0I_res
r2_res
s0II_res
rmspe_res



prep_shared_mem_with_name -> MOCK
curve_fit -> MOCK
close_shr_mem -> MOCK
get_bounds
get_max_iter
get_model
get_guesses
eval_model
"""
