
import abc
import numpy as np
from scipy.optimize import curve_fit
from tedana.workflows.sage import (
    concurrency_sage,
    loglinear_sage,
)


class GetMapsNonlinear(metaclass=abc.ABCMeta):
    def __init__(self, n_param):
        self.n_param = n_param
    def set_n_param(self, n_param):
        self.n_param = n_param
    @abc.abstractmethod
    def get_model(self, i_v, i_t, delta):
        return NotImplemented
    @abc.abstractmethod
    def get_guesses(self, i_v, i_t, arrs_shr_mem):
        return NotImplemented
    @abc.abstractmethod
    def get_bounds(self):
        return NotImplemented
    @abc.abstractmethod
    def get_max_iter(self):
        return NotImplemented
    @abc.abstractmethod
    def eval_model(self, i_v, i_t, X, arrs_shr_mem, model):
        return NotImplemented

    @staticmethod
    def get_normalized_guesses(data, tes, mask):
        t2star_guess, s0_I_guess, t2_guess, _, delta_guess, _ = loglinear_sage.get_maps_loglinear(
            data, tes, mask, -1
        )
        r2star_guess = 1 / t2star_guess
        r2_guess = 1 / t2_guess
        r2star_guess[~np.isfinite(r2star_guess)] = 20
        r2_guess[~np.isfinite(r2_guess)] = 15
        s0_I_guess[~np.isfinite(s0_I_guess)] = np.mean(s0_I_guess[np.isfinite(s0_I_guess)])
        delta_guess[np.logical_or(delta_guess < -9, delta_guess > 11)] = 1
        delta_guess[np.isnan(delta_guess)] = 1
        s0_II_guess = s0_I_guess / delta_guess
        s0_II_guess[~np.isfinite(s0_II_guess)] = np.mean(s0_II_guess[np.isfinite(s0_II_guess)])
        return r2star_guess, s0_I_guess, r2_guess, s0_II_guess, delta_guess

    @staticmethod
    def get_normalized_delta(s0_I, s0_II):
        delta = np.divide(s0_I, s0_II)
        delta[np.logical_or(delta < -9, delta > 11)] = 1
        delta[np.isnan(delta)] = 1
        return delta

    @staticmethod
    def init_maps(shape):
        r2star_res = np.zeros(shape)
        s0_I_res = np.zeros(shape)
        r2_res = np.zeros(shape)
        s0_II_res = np.zeros(shape)
        rmspe_res = np.zeros(shape)
        return r2star_res, s0_I_res, r2_res, s0_II_res, rmspe_res

    def fit_nonlinear_sage(
        self,
        shape,
        dtype,
        t_start,
        t_end,
        Y,
        X,
        r2star_guess,
        s0_I_guess,
        r2_guess,
        s0_II_guess,
        delta,
        r2star_res,
        s0_I_res,
        r2_res,
        s0_II_res,
        rmspe_res,
    ):
        shr_mem_X = {
            "X": X,
        }
        shr_mem_Y = {
            "Y": Y,
        }
        shr_mems = {
            "r2star_guess": r2star_guess,
            "s0_I_guess": s0_I_guess,
            "r2_guess": r2_guess,
            "s0_II_guess": s0_II_guess,
            "delta": delta,
            "r2star_res": r2star_res,
            "s0_I_res": s0_I_res,
            "r2_res": r2_res,
            "s0_II_res": s0_II_res,
            "rmspe_res": rmspe_res,
        }
        n_samps, n_echos, n_vols = shape
        shr_mem_X, arr_shr_mem_X = concurrency_sage.prep_shared_mem_with_name(
            shr_mem_X, (n_echos,), dtype
        )
        shr_mem_Y, arr_shr_mem_Y = concurrency_sage.prep_shared_mem_with_name(
            shr_mem_Y, (n_samps, n_echos, n_vols), dtype
        )
        X = arr_shr_mem_X["X"]
        Y = arr_shr_mem_Y["Y"]
        shr_mems, arrs_shr_mem = concurrency_sage.prep_shared_mem_with_name(
            shr_mems, (n_samps, n_vols), dtype
        )

        fail_count = 0
        bounds = self.get_bounds()
        max_iter = self.get_max_iter()
        for i_v in range(n_samps):
            for i_t in range(t_start, t_end):
                try:
                    model = self.get_model(i_v, i_t, arrs_shr_mem["delta"])
                    popt, _ = curve_fit(
                        model,
                        X,
                        Y[i_v, :, i_t],
                        p0=self.get_guesses(i_v, i_t, arrs_shr_mem),
                        bounds=bounds,
                        ftol=1e-12,
                        xtol=1e-12,
                        max_nfev=max_iter,
                    )
                    arrs_shr_mem["r2star_res"][i_v, i_t] = popt[0]
                    arrs_shr_mem["s0_I_res"][i_v, i_t] = popt[1]
                    arrs_shr_mem["r2_res"][i_v, i_t] = popt[2]
                    if arrs_shr_mem["s0_II_res"] is not None:
                        arrs_shr_mem["s0_II_res"][i_v, i_t] = popt[3]
                    arrs_shr_mem["rmspe_res"][i_v, i_t] = np.sqrt(
                        np.mean(
                            np.square(
                                (
                                    Y[i_v, :, i_t]
                                    - self.eval_model(i_v, i_t, X, arrs_shr_mem, model)
                                )
                                / Y[i_v, :, i_t]
                                * 100
                            )
                        )
                    )

                except (RuntimeError, ValueError):
                    fail_count += 1

        if fail_count:
            fail_percent = 100 * fail_count / (n_samps * (t_end - t_start))
            print("fail_percent: ", fail_percent)

        concurrency_sage.close_shr_mem(shr_mem_X)
        concurrency_sage.close_shr_mem(shr_mem_Y)
        concurrency_sage.close_shr_mem(shr_mems)
