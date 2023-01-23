import numpy as np
from tedana.workflows.sage import config_sage, concurrency_sage, utils_sage
from tedana.workflows.sage.nonlinear_sage import GetMapsNonlinear


class Get_Maps_Nonlinear_3Param(GetMapsNonlinear):
    def get_model(self, i_v, i_t, delta):
        if self.n_param == 3:
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
        elif self.n_param == 4:

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
        else:
            raise ValueError("Invalid value for number of parameters")

    def get_guesses(self, i_v, i_t, arrs_shr_mem):
        if self.n_param == 4:
            return (
                arrs_shr_mem["r2star_guess"][i_v, i_t],
                arrs_shr_mem["s0I_guess"][i_v, i_t],
                arrs_shr_mem["r2_guess"][i_v, i_t],
                arrs_shr_mem["s0II_guess"][i_v, i_t],
            )
        elif self.n_param == 3:
            return (
                arrs_shr_mem["r2star_guess"][i_v, i_t],
                arrs_shr_mem["s0I_guess"][i_v, i_t],
                arrs_shr_mem["r2_guess"][i_v, i_t],
            )
        else:
            raise ValueError("Invalid value for number of parameters")

    def eval_model(self, i_v, i_t, X, arrs_shr_mem, model):
        if self.n_param == 4:
            return model(
                X,
                arrs_shr_mem["r2star_res"][i_v, i_t],
                arrs_shr_mem["s0I_res"][i_v, i_t],
                arrs_shr_mem["r2_res"][i_v, i_t],
                arrs_shr_mem["s0II_res"][i_v, i_t],
            )
        elif self.n_param == 3:
            return model(
                X,
                arrs_shr_mem["r2star_res"][i_v, i_t],
                arrs_shr_mem["s0I_res"][i_v, i_t],
                arrs_shr_mem["r2_res"][i_v, i_t],
            )
        else:
            raise ValueError("Invalid value for number of parameters")

    def get_max_iter(self):
        if self.n_param == 3:
            return 10000
        elif self.n_param == 4:
            return 1000
        else:
            raise ValueError("Invalid value for number of parameters")

    def get_bounds(self):
        if self.n_param == 4:
            return (
                (0.1, 0, 0.1, 0),
                (10000, np.inf, 10000, np.inf),
            )
        elif self.n_param == 3:
            return (
                (0.1, 0, 0.1),
                (10000, np.inf, 10000),
            )
        else:
            raise ValueError("Invalid value for number of parameters")


def get_maps_nonlinear_3param(data, tes, mask, n_procs):
    """
    Performs 3-param fit for T2* and T2 for SAGE sequence
    after performing loose 4-param fit to find delta values.
    """
    nonlinear_fitter = Get_Maps_Nonlinear_3Param(n_param=4)

    r2star_res, s0I_res, r2_res, s0II_res, rmspe_res = utils_sage.init_arrs(
        config_sage.get_shape_maps(data), len(config_sage.get_keys_maps_nonlin_4param())
    )

    (
        r2star_guess,
        s0I_guess,
        r2_guess,
        s0II_guess,
        _,
    ) = GetMapsNonlinear.get_normalized_guesses(data, tes, mask)

    mask = mask.reshape(mask.shape[0])

    Y = data[mask, :, :]
    X = tes

    # these must match the function signature of fit_nonlinear_sage
    dict_shr_mem_4param = GetMapsNonlinear.get_dict_shr_mem_masked(
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

    shr_mems_4param, arrs_shr_mem_4param = concurrency_sage.prep_shared_mem_with_arr(
        dict_shr_mem_4param
    )

    kwargs_4param = {
        key: (value.name if value is not None else None) for key, value in shr_mems_4param.items()
    }

    shape = Y.shape
    dtype = Y.dtype
    n_iter = Y.shape[config_sage.get_dim_vols()]

    args_4param = (shape, dtype)
    func = nonlinear_fitter.fit_nonlinear_sage

    procs_4param = concurrency_sage.get_procs(n_iter, func, n_procs, args_4param, kwargs_4param)
    concurrency_sage.start_procs(procs_4param)
    concurrency_sage.join_procs(procs_4param)

    r2star_guess = arrs_shr_mem_4param["r2star_res"]
    s0I_guess = arrs_shr_mem_4param["s0I_res"]
    r2_guess = arrs_shr_mem_4param["r2_res"]
    delta_res = GetMapsNonlinear.get_normalized_delta(
        arrs_shr_mem_4param["s0I_res"], arrs_shr_mem_4param["s0II_res"]
    )

    r2star_res, s0I_res, r2_res, rmspe_res = utils_sage.init_arrs(
        config_sage.get_shape_maps(data),
        len(config_sage.get_keys_maps_results_nonlin_3param_short()),
    )

    shr_mem_keys_3param = GetMapsNonlinear.get_dict_shr_mem_masked(
        Y,
        X,
        mask,
        r2star_guess=None,
        s0I_guess=None,
        r2_guess=None,
        s0II_guess=None,
        delta_res=delta_res,
        r2star_res=r2star_res,
        s0I_res=s0I_res,
        r2_res=r2_res,
        s0II_res=None,
        rmspe_res=rmspe_res,
    )

    # exclude Y and X because they are already allocated
    shr_mems_3param, arrs_shr_mem_3param = concurrency_sage.prep_shared_mem_with_arr(
        dict(filter(lambda item: item[0] not in ["Y", "X"], shr_mem_keys_3param.items()))
    )

    kwargs_3param = {
        key: (value.name if value is not None else None) for key, value in shr_mems_3param.items()
    }
    kwargs_3param.update(
        {
            "Y": kwargs_4param["Y"],
            "X": kwargs_4param["X"],
            "r2star_guess": kwargs_4param["r2star_res"],
            "s0I_guess": kwargs_4param["s0I_res"],
            "r2_guess": kwargs_4param["r2_res"],
        }
    )
    args_3param = (shape, dtype)
    nonlinear_fitter.set_n_param(n_param=3)

    procs_3param = concurrency_sage.get_procs(n_iter, func, n_procs, args_3param, kwargs_3param)
    concurrency_sage.start_procs(procs_3param)
    concurrency_sage.join_procs(procs_3param)

    r2star_res, s0I_res, r2_res, delta_res, rmspe_res = utils_sage.unmask(
        list(
            filter(
                lambda val: True if val is not None else False,
                map(
                    lambda item: item[1]
                    if item[0] in config_sage.get_keys_maps_results_nonlin_3param()
                    else None,
                    arrs_shr_mem_3param.items(),
                ),
            ),
        ),
        mask,
    )

    concurrency_sage.close_and_unlink_shr_mem(shr_mems_4param)
    concurrency_sage.close_and_unlink_shr_mem(shr_mems_3param)

    t2star_res = 1 / r2star_res
    t2_res = 1 / r2_res
    s0II_res = s0I_res / delta_res

    return t2star_res, s0I_res, t2_res, s0II_res, delta_res, rmspe_res
