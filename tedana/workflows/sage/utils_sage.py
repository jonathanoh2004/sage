import numpy as np
import tedana.utils


def init_arrs(shape, num_arrs):
    """
    Convenience function used to allocate a list of the
    indicated number of zero arrays of the indicated shape
    ----- INPUT -----
        shape: tuple[int, ...]
        num_arrs: int
    ----- OUTPUT -----
        arrs: list[numpy.ndarray, ...]
    """
    arrs = []
    for _ in range(num_arrs):
        arrs.append(np.zeros(shape))
    return arrs


def unmask(arrs, mask):
    """
    Convenience function to unmask a list of arrays
    and check for None values.
    ----- INPUT -----
        arrs: list[numpy.ndarray | None]
    ----- OUTPUT -----
        res: list[numpy.ndarray, ...]
    """
    res = []
    for arr in arrs:
        if arr is not None:
            res.append(tedana.utils.unmask(arr, mask))
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
