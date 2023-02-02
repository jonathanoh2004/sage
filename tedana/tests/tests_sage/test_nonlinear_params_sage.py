from unittest.mock import patch
import numpy as np
import tedana.workflows.sage.nonlinear_3param_sage as nonlinear_3param
import tedana.workflows.sage.nonlinear_4param_sage as nonlinear_4param


@patch("tedana.workflows.sage.nonlinear_sage.GetMapsNonlinear.get_normalized_guesses")
@patch("tedana.workflows.sage.nonlinear_sage.GetMapsNonlinear.get_normalized_delta")
def test_get_maps_nonlinear_params(mock_normalized_delta, mock_normalized_guesses):
    r2star = np.log(np.ones((2, 2)) * 125)
    r2 = np.log(np.ones((2, 2)) * 5)
    s0I = np.ones((2, 2)) * 10
    s0II = np.ones((2, 2)) * 10
    err = 2e-13
    mock_normalized_delta.side_effect = lambda s0I, s0II: np.ones((2, 2), dtype=np.float64)
    mock_normalized_guesses.side_effect = lambda data, tes, mask: [
        r2star + err,
        s0I + err,
        r2 + err,
        s0II + err,
        None,
    ]
    data = (
        np.array([0.08, 0.08, 0.0032, 0.0032, 0.0032])[np.newaxis, :, np.newaxis]
        .repeat(2, axis=0)
        .repeat(2, axis=2)
    )
    tes = np.array([1, 1, 5, 5, 5], dtype=np.float64)
    mask = np.array([1, 1]).astype(bool)
    n_procs = 2

    t2star_exp = 1 / r2star
    t2_exp = 1 / r2
    s0I_exp = s0I
    s0II_exp = s0II
    delta_exp = s0I_exp / s0II_exp

    rtol = 1e-12
    rtol_s0 = 1e-10
    tol_rmspe = 1e-10

    for nonlinear_get_maps in [
        nonlinear_3param.get_maps_nonlinear_3param,
        nonlinear_4param.get_maps_nonlinear_4param,
    ]:
        (
            t2star_test,
            s0I_test,
            t2_test,
            s0II_test,
            delta_test,
            rmspe_test,
        ) = nonlinear_get_maps(data, tes, mask, n_procs)

        np.testing.assert_allclose(s0I_test, s0I_exp, rtol=rtol_s0)
        np.testing.assert_allclose(t2star_test, t2star_exp, rtol=rtol)
        np.testing.assert_allclose(s0II_test, s0II_exp, rtol=rtol_s0)
        np.testing.assert_allclose(t2_test, t2_exp, rtol=rtol)
        np.testing.assert_allclose(delta_test, delta_exp, rtol=rtol)
        assert np.all(rmspe_test < tol_rmspe)
