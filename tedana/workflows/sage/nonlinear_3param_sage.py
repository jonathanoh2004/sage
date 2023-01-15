import numpy as np
import config_sage
import concurrency_sage
import nonlinear_sage
import utils_sage


def _get_model(i_v, i_t, delta=None):
    if delta is not None:
        delta = delta[i_v, i_t]

        def _three_param(X, r2star, s0_I, r2):
            res = np.zeros(X.shape)

            res[X < X[-1] / 2] = s0_I * np.exp(-1 * X[X < X[-1] / 2] * r2star)
            res[X > X[-1] / 2] = (
                (1 / delta)
                * np.exp(-1 * X[-1] * (r2star - r2))
                * np.exp(-1 * X[X > X[-1] / 2] * ((2 * r2) - r2star))
                * (s0_I)
            )
            return res

        return _three_param
    else:

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


def _get_max_iter(fittype, arrs_shr_mem):
    if fittype == "nonlin3":
        if arrs_shr_mem["s0_II_guess"] is None:
            return 10000
        else:
            return 1000
    else:
        return 10000


def _get_guesses(i_v, i_t, arrs_shr_mem):
    if arrs_shr_mem["s0_II_guess"] is not None:
        return (
            arrs_shr_mem["r2star_guess"][i_v, i_t],
            arrs_shr_mem["s0_I_guess"][i_v, i_t],
            arrs_shr_mem["r2_guess"][i_v, i_t],
            arrs_shr_mem["s0_II_guess"][i_v, i_t],
        )
    else:
        return (
            arrs_shr_mem["r2star_guess"][i_v, i_t],
            arrs_shr_mem["s0_I_guess"][i_v, i_t],
            arrs_shr_mem["r2_guess"][i_v, i_t],
        )


def _eval_model(i_v, i_t, X, arrs_shr_mem, model):
    if arrs_shr_mem["s0_II_guess"] is not None:
        return model(
            X,
            arrs_shr_mem["r2star_res"][i_v, i_t],
            arrs_shr_mem["s0_I_res"][i_v, i_t],
            arrs_shr_mem["r2_res"][i_v, i_t],
            arrs_shr_mem["s0_II_res"][i_v, i_t],
        )
    else:
        return model(
            X,
            arrs_shr_mem["r2star_res"][i_v, i_t],
            arrs_shr_mem["s0_I_res"][i_v, i_t],
            arrs_shr_mem["r2_res"][i_v, i_t],
        )


def _get_bounds(arrs_shr_mem):
    if arrs_shr_mem["s0_II_guess"] is not None:
        return (
            (0.1, 0, 0.1, 0),
            (10000, np.inf, 10000, np.inf),
        )
    else:
        return (
            (0.1, 0, 0.1),
            (10000, np.inf, 10000),
        )


def get_maps_nonlinear_3param(data, tes, mask):
    n_samps, n_echos, n_vols = (
        config_sage.get_n_samps(data),
        config_sage.get_n_echos(data),
        config_sage.get_n_vols(data),
    )
    r2star_res, s0_I_res, r2_res, s0_II_res, rmspe_res = nonlinear_sage.init_maps(n_samps, n_vols)

    r2star_guess, s0_I_guess, r2_guess, s0_II_guess, delta = nonlinear_sage.get_guesses(
        data, tes, mask
    )

    mask = mask.reshape(mask.shape[0])

    Y = data[mask, :, :]
    X = tes

    # these must match the function signature of fit_nonlinear_sage
    shr_mem_keys_4param = dict(
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

    shr_mems_4param, arrs_shr_mem_4param = concurrency_sage.prep_shared_mem(shr_mem_keys_4param)

    kwargs = {
        key: (value.name if value is not None else None) for key, value in shr_mems_4param.items()
    }

    dim_iter = config_sage.get_dim_vols()
    shape = data.shape
    func = nonlinear_sage.get_fit_nonlinear_sage(
        _get_model, _get_max_iter, _get_guesses, _eval_model, _get_bounds
    )
    args = (Y.shape[0], n_echos, n_vols, data.dtype)

    procs_4param = concurrency_sage.get_procs(shape, dim_iter, func, args, kwargs)
    concurrency_sage.start_procs(procs_4param)
    concurrency_sage.join_procs(procs_4param)

    r2star_guess = arrs_shr_mem_4param["r2star_res"]
    s0_I_guess = arrs_shr_mem_4param["s0_I_res"]
    r2_guess = arrs_shr_mem_4param["r2_res"]
    delta = nonlinear_sage.get_normalized_delta(arrs_shr_mem_4param)

    r2star_res = np.zeros((n_samps, n_vols))
    s0_I_res = np.zeros((n_samps, n_vols))
    r2_res = np.zeros((n_samps, n_vols))
    rmspe_res = np.zeros((n_samps, n_vols))

    shr_mem_keys_3param = dict(
        zip(
            config_sage.get_nonlinear_keys(),
            [
                Y,
                X,
                None,
                None,
                None,
                None,
                delta,
                r2star_res[mask, :],
                s0_I_res[mask, :],
                r2_res[mask, :],
                None,
                rmspe_res[mask, :],
            ],
        )
    )

    shr_mems_3param, arrs_shr_mem_3param = concurrency_sage.prep_shared_mem(shr_mem_keys_3param)

    kwargs_3param = {
        key: (value.name if value is not None else None) for key, value in shr_mems_3param.items()
    }
    kwargs_3param.update(
        {
            "Y": kwargs["Y"],
            "X": kwargs["X"],
            "r2star_guess": kwargs["r2star_res"],
            "s0_I_guess": kwargs["s0_I_res"],
            "r2_guess": kwargs["r2_res"],
        }
    )

    procs_3param = concurrency_sage.get_procs(shape, dim_iter, func, args, kwargs)
    concurrency_sage.start_procs(procs_3param)
    concurrency_sage.join_procs(procs_3param)

    r2star_res, s0_I_res, r2_res, s0_II_res, delta, rmspe_res = utils_sage._unmask_and_copy(
        arrs_shr_mem_3param, mask
    )

    for shm in shr_mems_3param.values():
        if shm is not None:
            shm.close()
            shm.unlink()

    r2star_res, s0_I_res, r2_res, s0_II_res, delta, rmspe_res = utils_sage._unmask_and_copy(
        arrs_shr_mem_4param, mask
    )

    for shm in shr_mems_4param.values():
        if shm is not None:
            shm.close()
            shm.unlink()

    t2star_res = 1 / r2star_res
    t2_res = 1 / r2_res

    return t2star_res, s0_I_res, t2_res, s0_II_res, delta, rmspe_res
