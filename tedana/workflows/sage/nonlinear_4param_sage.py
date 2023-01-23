import numpy as np
from tedana.workflows.sage import concurrency_sage, config_sage, utils_sage
from tedana.workflows.sage.nonlinear_sage import GetMapsNonlinear


class Get_Maps_Nonlinear_4Param(GetMapsNonlinear):
    def get_model(self, i_v, i_t, delta):
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

    def get_guesses(self, i_v, i_t, arrs_shr_mem):
        return (
            arrs_shr_mem["r2star_guess"][i_v, i_t],
            arrs_shr_mem["s0I_guess"][i_v, i_t],
            arrs_shr_mem["r2_guess"][i_v, i_t],
            arrs_shr_mem["s0II_guess"][i_v, i_t],
        )

    def get_bounds(self):
        return (
            (0.1, 0, 0.1, 0),
            (10000, np.inf, 10000, np.inf),
        )

    def get_max_iter(self):
        return 10000

    def eval_model(self, i_v, i_t, X, arrs_shr_mem, model):
        return model(
            X,
            arrs_shr_mem["r2star_res"][i_v, i_t],
            arrs_shr_mem["s0I_res"][i_v, i_t],
            arrs_shr_mem["r2_res"][i_v, i_t],
            arrs_shr_mem["s0II_res"][i_v, i_t],
        )


def get_maps_nonlinear_4param(data, tes, mask, n_procs):
    nonlinear_fitter = Get_Maps_Nonlinear_4Param(n_param=4)

    r2star_res, s0I_res, r2_res, s0II_res, rmspe_res = utils_sage.init_arrs(
        config_sage.get_shape_maps(data), len(config_sage.get_keys_maps_nonlin_4param())
    )

    r2star_guess, s0I_guess, r2_guess, s0II_guess, _ = GetMapsNonlinear.get_normalized_guesses(
        data, tes, mask
    )

    mask = mask.reshape(mask.shape[0])

    Y = data[mask, :, :]
    X = tes

    # these must match the function signature of fit_nonlinear_sage
    dict_shr_mem = GetMapsNonlinear.get_dict_shr_mem_masked(
        Y,
        X,
        mask,
        r2star_guess=r2star_guess,
        s0I_guess=s0I_guess,
        r2_guess=r2_guess,
        s0II_guess=s0II_guess,
        delta_res=None,
        r2star_res=r2star_res,
        s0I_res=s0I_res,
        r2_res=r2_res,
        s0II_res=s0II_res,
        rmspe_res=rmspe_res,
    )

    shr_mems, arrs_shr_mem = concurrency_sage.prep_shared_mem_with_arr(dict_shr_mem)

    kwargs = {key: (value.name if value is not None else None) for key, value in shr_mems.items()}

    shape = Y.shape
    dtype = Y.dtype
    n_iter = Y.shape[config_sage.get_dim_vols()]

    args = (shape, dtype)
    func = nonlinear_fitter.fit_nonlinear_sage

    procs = concurrency_sage.get_procs(n_iter, func, n_procs, args, kwargs)
    concurrency_sage.start_procs(procs)
    concurrency_sage.join_procs(procs)

    r2star_res, s0I_res, r2_res, s0II_res, rmspe_res = utils_sage.unmask_and_copy(
        list(
            filter(
                lambda val: True if val is not None else False,
                map(
                    lambda item: item[1]
                    if item[0] in config_sage.get_keys_maps_results_nonlin_4param()
                    else None,
                    arrs_shr_mem.items(),
                ),
            ),
        ),
        mask,
    )

    concurrency_sage.close_and_unlink_shr_mem(shr_mems)

    t2star_res = 1 / r2star_res
    t2_res = 1 / r2_res
    delta_res = s0I_res / s0II_res

    return t2star_res, s0I_res, t2_res, s0II_res, delta_res, rmspe_res
