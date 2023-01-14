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


def fit_decay_sage(data, tes, mask, fittype):
    n_samps, n_echos, n_vols = data.shape
    dtype = data.dtype

    if fittype == "loglin":
        fit_loglinear_sage(data, tes, mask)

    elif fittype == "nonlin4" or fittype == "nonlin3":
        guesses = get_guesses()
        fit_nonlinear_sage(data, tes, mask, guesses)
    else:
        raise ValueError("invalid fittype")


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
