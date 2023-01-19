import tedana.utils
from tedana.workflows.sage import config_sage


def chain(funcs, args, kwargs):
    pass


def unmask_and_copy(arrs_shr_mem, mask):
    res = []
    valid_keys = config_sage.get_nonlinear_result_keys()
    for key in arrs_shr_mem:
        if key in valid_keys:
            res.append(tedana.utils.unmask(arrs_shr_mem[key], mask).copy())
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
