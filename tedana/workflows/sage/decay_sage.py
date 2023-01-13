"""
Functions to estimate S0 and T2* from multi-echo data.
"""
import logging
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import scipy

from tedana import utils


LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

######################################################################################
################################## SAGE TEDANA #######################################
######################################################################################
def _prep_shared_mem(mapping):
    """
    Takes in numpy arrays and corresponding keys
    Outputs one dictionary mapping from keys to shared memory names
    as well as a second dictionary mapping from keys to numpy arrays
    referring to same underlying memory
    """
    if not isinstance(mapping, dict):
        raise ValueError("input must be a dictionary mapping from names to shared memory objects")
    shr_mems, arrs_shr_mem = {}, {}

    for (key, arr) in mapping.items():
        if arr is not None:
            shm = SharedMemory(create=True, size=arr.nbytes)
            arr_shr_mem = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            arr_shr_mem[:] = arr[:]
            shr_mems[key] = shm
            arrs_shr_mem[key] = arr_shr_mem
        else:
            shr_mems[key] = None

    return shr_mems, arrs_shr_mem


def _start_and_join_procs(n_samps, n_echos, n_vols, dtype, fittype, kwargs):
    n_cpus = multiprocessing.cpu_count()

    procs = []
    n_tasks_per_cpu = max(n_vols // n_cpus, 1)

    i_cpu, t_start, t_end = 0, 0, 0
    while i_cpu < n_cpus and t_end < n_vols:
        if i_cpu + 1 >= n_cpus or t_start + n_tasks_per_cpu > n_vols:
            t_end = n_vols
        else:
            t_end = t_start + n_tasks_per_cpu
        proc = multiprocessing.Process(
            target=fit_nonlinear_sage,
            args=(n_samps, n_echos, n_vols, dtype, t_start, t_end, fittype),
            kwargs=kwargs,
        )
        procs.append(proc)
        proc.start()
        t_start = t_end
        i_cpu += 1

    for proc in procs:
        proc.join()


def fit_loglinear_sage(data, tes, mask):
    n_samps, n_echos, n_vols = data.shape
    dtype = data.dtype

    n_samps, n_echos, n_vols = data.shape
    tese = tes[-1]

    Y = data.swapaxes(1, 2).reshape((n_samps * n_vols, -1)).T
    Y = np.log(Y) * (np.repeat(mask, axis=0, repeats=n_vols).T)

    x_s0_I = np.ones(n_echos)
    x_delta = np.array([0, 0, -1, -1, -1])
    x_r2star = np.array(
        [
            -1 * tes[0],
            -1 * tes[1],
            tes[2] - tese,
            tes[3] - tese,
            0,
        ]
    )
    x_r2 = np.array([0, 0, tese - (2 * tes[2]), tese - (2 * tes[3]), -1 * tese])

    X = np.column_stack([x_s0_I, x_delta, x_r2star, x_r2])

    Y[~np.isfinite(Y)] = 0

    betas = np.linalg.lstsq(X, Y, rcond=None)[0]

    betas[~np.isfinite(betas)] = 0

    s0_I_map = np.exp(betas[0, :]).T
    delta_map = np.exp(betas[1, :]).T
    s0_II_map = s0_I_map / delta_map
    t2star_map = 1 / betas[2, :].T
    t2_map = 1 / betas[3, :].T

    if n_vols > 1:
        s0_I_map = s0_I_map.reshape(n_samps, n_vols)
        s0_II_map = s0_II_map.reshape(n_samps, n_vols)
        delta_map = delta_map.reshape(n_samps, n_vols)
        t2star_map = t2star_map.reshape(n_samps, n_vols)
        t2_map = t2_map.reshape(n_samps, n_vols)

    return t2star_map, s0_I_map, t2_map, s0_II_map, delta_map, None


def fit_decay_sage(data, tes, mask, fittype):
    n_samps, n_echos, n_vols = data.shape
    dtype = data.dtype

    if fittype == "loglin":
        fit_loglinear_sage(data, tes, mask)

    elif fittype == "nonlin4" or fittype == "nonlin3":
        r2star_res = np.zeros((n_samps, n_vols))
        s0_I_res = np.zeros((n_samps, n_vols))
        r2_res = np.zeros((n_samps, n_vols))
        s0_II_res = np.zeros((n_samps, n_vols))
        rmspe_res = np.zeros((n_samps, n_vols))

        t2star_guess, s0_I_guess, t2_guess, _, delta_guess, _ = fit_loglinear_sage(data, tes, mask)
        r2star_guess = 1 / t2star_guess
        r2_guess = 1 / t2_guess
        r2star_guess[~np.isfinite(r2star_guess)] = 20
        r2_guess[~np.isfinite(r2_guess)] = 15
        s0_I_guess[~np.isfinite(s0_I_guess)] = np.mean(s0_I_guess[np.isfinite(s0_I_guess)])
        delta_guess[np.logical_or(delta_guess < -9, delta_guess > 11)] = 1
        delta_guess[np.isnan(delta_guess)] = 1
        s0_II_guess = s0_I_guess / delta_guess
        s0_II_guess[~np.isfinite(s0_II_guess)] = np.mean(s0_II_guess[np.isfinite(s0_II_guess)])

        mask = mask.reshape(mask.shape[0])

        Y = data[mask, :, :]
        X = tes

        # these must match the function signature of fit_nonlinear_sage
        shr_mem_keys = {
            "Y": Y,
            "X": X,
            "r2star_guess": r2star_guess[mask, :],
            "s0_I_guess": s0_I_guess[mask, :],
            "r2_guess": r2_guess[mask, :],
            "s0_II_guess": s0_II_guess[mask, :],
            "delta": None,
            "r2star_res": r2star_res[mask, :],
            "s0_I_res": s0_I_res[mask, :],
            "r2_res": r2_res[mask, :],
            "s0_II_res": s0_II_res[mask, :],
            "rmspe_res": rmspe_res[mask, :],
        }
        postfix_kwargs_shm = (
            "_shr_name"  # must correspond to the argument names in nonlinear fitting function
        )

        shr_mems, arrs_shr_mem = _prep_shared_mem(shr_mem_keys)

        kwargs_shm = {
            key + postfix_kwargs_shm: (value.name if value is not None else None)
            for key, value in shr_mems.items()
        }

        # max_iter = niter3[0] if niter3 is not None else 10000

        _start_and_join_procs(Y.shape[0], n_echos, n_vols, dtype, fittype, kwargs_shm)

        if fittype == "nonlin3":
            r2star_guess = arrs_shr_mem["r2star_res"]
            s0_I_guess = arrs_shr_mem["s0_I_res"]
            r2_guess = arrs_shr_mem["r2_res"]
            delta = np.divide(arrs_shr_mem["s0_I_res"], arrs_shr_mem["s0_II_res"])
            delta[np.logical_or(delta < -9, delta > 11)] = 1
            delta[np.isnan(delta)] = 1

            r2star_res = np.zeros((n_samps, n_vols))
            s0_I_res = np.zeros((n_samps, n_vols))
            r2_res = np.zeros((n_samps, n_vols))
            rmspe_res = np.zeros((n_samps, n_vols))

            shr_mem_keys_3param = {
                "Y": None,
                "X": None,
                "r2star_guess": None,
                "s0_I_guess": None,
                "r2_guess": None,
                "s0_II_guess": None,
                "delta": delta,
                "r2star_res": r2star_res[mask, :],
                "s0_I_res": s0_I_res[mask, :],
                "r2_res": r2_res[mask, :],
                "s0_II_res": None,
                "rmspe_res": rmspe_res[mask, :],
            }

            shr_mems_3param, arrs_shr_mem_3param = _prep_shared_mem(shr_mem_keys_3param)

            kwargs_shm_3param = {
                key + postfix_kwargs_shm: (value.name if value is not None else None)
                for key, value in shr_mems_3param.items()
            }
            kwargs_shm_3param.update(
                {
                    "Y" + postfix_kwargs_shm: kwargs_shm["Y" + postfix_kwargs_shm],
                    "X" + postfix_kwargs_shm: kwargs_shm["X" + postfix_kwargs_shm],
                    "r2star_guess"
                    + postfix_kwargs_shm: kwargs_shm["r2star_res" + postfix_kwargs_shm],
                    "s0_I_guess" + postfix_kwargs_shm: kwargs_shm["s0_I_res" + postfix_kwargs_shm],
                    "r2_guess" + postfix_kwargs_shm: kwargs_shm["r2_res" + postfix_kwargs_shm],
                }
            )

            _start_and_join_procs(Y.shape[0], n_echos, n_vols, dtype, fittype, kwargs_shm_3param)

            r2star_res = utils.unmask(arrs_shr_mem_3param["r2star_res"], mask).copy()
            s0_I_res = utils.unmask(arrs_shr_mem_3param["s0_I_res"], mask).copy()
            r2_res = utils.unmask(arrs_shr_mem_3param["r2_res"], mask).copy()
            delta = utils.unmask(arrs_shr_mem_3param["delta"], mask).copy()
            rmspe_res = utils.unmask(arrs_shr_mem_3param["rmspe_res"], mask).copy()
            s0_II_res = s0_I_res / delta

            for shm in shr_mems_3param.values():
                if shm is not None:
                    shm.close()
                    shm.unlink()

        elif fittype == "nonlin4":
            r2star_res = utils.unmask(arrs_shr_mem["r2star_res"], mask).copy()
            s0_I_res = utils.unmask(arrs_shr_mem["s0_I_res"], mask).copy()
            r2_res = utils.unmask(arrs_shr_mem["r2_res"], mask).copy()
            s0_II_res = utils.unmask(arrs_shr_mem["s0_II_res"], mask).copy()
            rmspe_res = utils.unmask(arrs_shr_mem["rmspe_res"], mask).copy()
            delta = s0_I_res / s0_II_res
        else:
            raise ValueError("invalid fittype")

        for shm in shr_mems.values():
            if shm is not None:
                shm.close()
                shm.unlink()

        t2star_res = 1 / r2star_res
        t2_res = 1 / r2_res

        return t2star_res, s0_I_res, t2_res, s0_II_res, delta, rmspe_res
    else:
        raise ValueError("invalid fittype")


def fit_loglinear_sage(data_cat, echo_times, mask):
    """
    This function fits over each voxel independently (all time points)
    """


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


def fit_nonlinear_sage(
    n_samps,
    n_echos,
    n_vols,
    dtype,
    t_start,
    t_end,
    fittype,
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
    max_iter = _get_max_iter(fittype, arrs_shr_mem)
    for i_v in range(n_samps):
        for i_t in range(t_start, t_end):
            try:
                model = _get_model(i_v, i_t, arrs_shr_mem["delta"])
                popt, _ = scipy.optimize.curve_fit(
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


######################################################################################
################################ UTILITY FUNCTION ####################################
######################################################################################


def _apply_t2s_floor(t2s, echo_times):
    """
    Apply a floor to T2* values to prevent zero division errors during
    optimal combination.

    Parameters
    ----------
    t2s : (S,) array_like
        T2* estimates.
    echo_times : (E,) array_like
        Echo times in milliseconds.

    Returns
    -------
    t2s_corrected : (S,) array_like
        T2* estimates with very small, positive values replaced with a floor value.
    """
    t2s_corrected = t2s.copy()
    echo_times = np.asarray(echo_times)
    if echo_times.ndim == 1:
        echo_times = echo_times[:, None]

    eps = np.finfo(dtype=t2s.dtype).eps  # smallest value for datatype
    nonzerovox = t2s != 0
    # Exclude values where t2s is 0 when dividing by t2s.
    # These voxels are also excluded from bad_voxel_idx
    temp_arr = np.zeros((len(echo_times), len(t2s)))
    temp_arr[:, nonzerovox] = np.exp(-echo_times / t2s[nonzerovox])  # (E x V) array
    bad_voxel_idx = np.any(temp_arr == 0, axis=0) & (t2s != 0)
    n_bad_voxels = np.sum(bad_voxel_idx)
    if n_bad_voxels > 0:
        n_voxels = temp_arr.size
        floor_percent = 100 * n_bad_voxels / n_voxels
        LGR.debug(
            "T2* values for {0}/{1} voxels ({2:.2f}%) have been "
            "identified as close to zero and have been "
            "adjusted".format(n_bad_voxels, n_voxels, floor_percent)
        )
    t2s_corrected[bad_voxel_idx] = np.min(-echo_times) / np.log(eps)
    return t2s_corrected
