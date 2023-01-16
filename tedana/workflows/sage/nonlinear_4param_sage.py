import numpy as np
import concurrency_sage
import config_sage
import nonlinear_sage
import utils_sage


def _get_model(n_param, i_v, i_t, delta):
    def _four_param(X, r2star, s0_I, r2, s0_II):
        res = np.zeros(X.shape)

        res[X < X[-1] / 2] = s0_I * np.exp(-1 * X[X < X[-1] / 2] * r2star)
        res[X > X[-1] / 2] = (
            s0_II
            * np.exp(-1 * X[-1] * (r2star - r2))
            * np.exp(-1 * X[X > X[-1] / 2] * ((2 * r2) - r2star))
        )
        return res

    return _four_param


def _get_max_iter(n_param):
    return 10000


def _get_guesses(n_param, i_v, i_t, arrs_shr_mem):
    return (
        arrs_shr_mem["r2star_guess"][i_v, i_t],
        arrs_shr_mem["s0_I_guess"][i_v, i_t],
        arrs_shr_mem["r2_guess"][i_v, i_t],
        arrs_shr_mem["s0_II_guess"][i_v, i_t],
    )


def _eval_model(n_param, i_v, i_t, X, arrs_shr_mem, model):
    return model(
        X,
        arrs_shr_mem["r2star_res"][i_v, i_t],
        arrs_shr_mem["s0_I_res"][i_v, i_t],
        arrs_shr_mem["r2_res"][i_v, i_t],
        arrs_shr_mem["s0_II_res"][i_v, i_t],
    )


def _get_bounds(n_param):
    return (
        (0.1, 0, 0.1, 0),
        (10000, np.inf, 10000, np.inf),
    )


def get_maps_nonlinear_4param(data, tes, mask):
    n_samps, n_echos, n_vols = (
        config_sage.get_n_samps(data),
        config_sage.get_n_echos(data),
        config_sage.get_n_vols(data),
    )
    r2star_res, s0_I_res, r2_res, s0_II_res, rmspe_res = nonlinear_sage.init_maps(
        (n_samps, n_vols)
    )

    r2star_guess, s0_I_guess, r2_guess, s0_II_guess, _ = nonlinear_sage.get_guesses(
        data, tes, mask
    )

    mask = mask.reshape(mask.shape[0])

    Y = data[mask, :, :]
    X = tes

    # these must match the function signature of fit_nonlinear_sage
    shr_mem_keys = dict(
        zip(
            config_sage.get_nonlinear_keys(),
            [
                Y,
                X,
                r2star_guess[mask, :],
                s0_I_guess[mask, :],
                r2_guess[mask, :],
                s0_II_guess[mask, :],
                None,
                r2star_res[mask, :],
                s0_I_res[mask, :],
                r2_res[mask, :],
                s0_II_res[mask, :],
                rmspe_res[mask, :],
            ],
        )
    )

    shr_mems, arrs_shr_mem = concurrency_sage.prep_shared_mem_with_arr(shr_mem_keys)

    kwargs = {key: (value.name if value is not None else None) for key, value in shr_mems.items()}

    dim_iter = config_sage.get_dim_vols()
    shape = data.shape
    n_param = 4
    func = nonlinear_sage.get_fit_nonlinear_sage(
        n_param, _get_model, _get_max_iter, _get_guesses, _eval_model, _get_bounds
    )
    args = (Y.shape[0], n_echos, n_vols, data.dtype)

    procs = concurrency_sage.get_procs(shape, dim_iter, func, args, kwargs)
    concurrency_sage.start_procs(procs)
    concurrency_sage.join_procs(procs)

    r2star_res, s0_I_res, r2_res, s0_II_res, delta, rmspe_res = utils_sage.unmask_and_copy(
        arrs_shr_mem, mask
    )

    concurrency_sage.close_and_unlink_shr_mem(shr_mems)

    t2star_res = 1 / r2star_res
    t2_res = 1 / r2_res

    return t2star_res, s0_I_res, t2_res, s0_II_res, delta, rmspe_res
