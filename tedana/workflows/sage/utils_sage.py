import numpy as np
import tedana.utils
from tedana.workflows.sage import config_sage


def init_arrs(shape, num_arrs):
    arrs = []
    for _ in range(num_arrs):
        arrs.append(np.zeros(shape))
    return arrs


def unmask_and_copy(arrs, mask):
    res = []
    for arr in arrs:
        if arr is not None:
            res.append(tedana.utils.unmask(arr, mask).copy())
    return res


def setup_loggers(repname, quiet, debug):
    tedana.utils.setup_loggers(logname=None, repname=repname, quiet=quiet, debug=debug)


def teardown_loggers():
    tedana.utils.teardown_loggers()


""" SCRAP
r2star_res = utils.unmask(arrs_shr_mem["r2star_res"], mask).copy()
    s0_I_res = utils.unmask(arrs_shr_mem["s0_I_res"], mask).copy()
    r2_res = utils.unmask(arrs_shr_mem["r2_res"], mask).copy()
    s0_II_res = utils.unmask(arrs_shr_mem["s0_II_res"], mask).copy()
    rmspe_res = utils.unmask(arrs_shr_mem["rmspe_res"], mask).copy()
    delta = s0_I_res / s0_II_res
    return r2star_res, s0_I_res, r2_res, s0_II_res, delta, rmspe_res
"""
