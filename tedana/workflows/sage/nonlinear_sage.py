import numpy as np
from multiprocessing.shared_memory import SharedMemory
import config_sage
import loglinear_sage
from scipy.optimize import curve_fit


def get_guesses(data, tes, mask):
    t2star_guess, s0_I_guess, t2_guess, _, delta_guess, _ = loglinear_sage.get_maps_loglinear(
        data, tes, mask
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


def get_normalized_delta(arrs_shr_mem_4param):
    delta = np.divide(arrs_shr_mem_4param["s0_I_res"], arrs_shr_mem_4param["s0_II_res"])
    delta[np.logical_or(delta < -9, delta > 11)] = 1
    delta[np.isnan(delta)] = 1
    return delta


def init_maps(n_samps, n_vols):
    r2star_res = np.zeros((n_samps, n_vols))
    s0_I_res = np.zeros((n_samps, n_vols))
    r2_res = np.zeros((n_samps, n_vols))
    s0_II_res = np.zeros((n_samps, n_vols))
    rmspe_res = np.zeros((n_samps, n_vols))
    return r2star_res, s0_I_res, r2_res, s0_II_res, rmspe_res


def get_fit_nonlinear_sage(model_func, max_iter_func, guesses_func, eval_model_func, bounds_func):
    _get_model = model_func
    _get_max_iter = max_iter_func
    _get_guesses = guesses_func
    _eval_model = eval_model_func
    _get_bounds = bounds_func

    def fit_nonlinear_sage(
        n_samps,
        n_echos,
        n_vols,
        dtype,
        t_start,
        t_end,
        X_shr_name,
        Y_shr_name,
        r2star_guess_shr_name,
        s0_I_guess_shr_name,
        r2_guess_shr_name,
        s0_II_guess_shr_name,
        delta_shr_name,
        r2star_res_shr_name,
        s0_I_res_shr_name,
        r2_res_shr_name,
        s0_II_res_shr_name,
        rmspe_res_shr_name,
    ):
        if s0_II_guess_shr_name is not None and delta_shr_name is not None:
            raise ValueError("at most one of s0_II and delta may be specified")

        shr_mems = {
            "X": X_shr_name,
            "Y": Y_shr_name,
            "r2star_guess": r2star_guess_shr_name,
            "s0_I_guess": s0_I_guess_shr_name,
            "r2_guess": r2_guess_shr_name,
            "s0_II_guess": s0_II_guess_shr_name,
            "delta": delta_shr_name,
            "r2star_res": r2star_res_shr_name,
            "s0_I_res": s0_I_res_shr_name,
            "r2_res": r2_res_shr_name,
            "s0_II_res": s0_II_res_shr_name,
            "rmspe_res": rmspe_res_shr_name,
        }
        arrs_shr_mem = {}

        for key in shr_mems:
            if shr_mems[key] is not None:
                shr_mems[key] = SharedMemory(name=shr_mems[key])

        X = np.ndarray(
            (n_echos),
            dtype=dtype,
            buffer=shr_mems["X"].buf,
        )
        Y = np.ndarray(
            (n_samps, n_echos, n_vols),
            dtype=dtype,
            buffer=shr_mems["Y"].buf,
        )
        for key in shr_mems:
            if shr_mems[key] is None:
                arrs_shr_mem[key] = None
            else:
                if key != "X" and key != "Y":
                    arrs_shr_mem[key] = np.ndarray(
                        (n_samps, n_vols),
                        dtype=dtype,
                        buffer=shr_mems[key].buf,
                    )

        fail_count = 0
        bounds = _get_bounds(arrs_shr_mem)
        max_iter = _get_max_iter(arrs_shr_mem)
        for i_v in range(n_samps):
            for i_t in range(t_start, t_end):
                try:
                    model = _get_model(i_v, i_t, arrs_shr_mem["delta"])
                    popt, _ = curve_fit(
                        model,
                        X,
                        Y[i_v, :, i_t],
                        p0=_get_guesses(i_v, i_t, arrs_shr_mem),
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
                                (Y[i_v, :, i_t] - _eval_model(i_v, i_t, X, arrs_shr_mem, model))
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

        for shr_mem in shr_mems.values():
            if shr_mem is not None:
                shr_mem.close()

    return fit_nonlinear_sage
