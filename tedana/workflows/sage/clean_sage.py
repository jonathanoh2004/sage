import numpy as np
import scipy.stats
from tedana.workflows.sage import utils_sage


def clean_maps_tedana(tes, maps_t2star, maps_t2, maps_s0I, maps_s0II):
    maps_t2star[np.isinf(maps_t2star)] = 500.0  # why 500?
    maps_t2star[maps_t2star <= 0] = 1.0  # let's get rid of negative values!
    maps_t2[np.isinf(maps_t2)] = 500.0  # why 500?
    maps_t2[maps_t2 <= 0] = 1.0  # let's get rid of negative values!
    # maps_t2star = apply_tedana_t2star_floor(maps_t2star, tes)
    maps_t2star, maps_t2 = utils_sage.apply_t2s_floor(maps_t2star, maps_t2, tes)
    maps_s0I[np.isnan(maps_s0I)] = 0.0  # why 0?
    maps_s0II[np.isnan(maps_s0II)] = 0.0  # why 0?
    cap_t2star = scipy.stats.scoreatpercentile(
        maps_t2star.ravel(), 99.5, interpolation_method="lower"
    )
    cap_t2 = scipy.stats.scoreatpercentile(maps_t2.ravel(), 99.5, interpolation_method="lower")
    maps_t2star[maps_t2star > cap_t2star * 10] = cap_t2star
    maps_t2[maps_t2 > cap_t2 * 10] = cap_t2


def clean_optcoms(optcom_t2star, optcom_t2):
    optcom_t2star[~np.isfinite(optcom_t2star)] = 0
    optcom_t2[~np.isfinite(optcom_t2)] = 0
