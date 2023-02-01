import logging
import numpy as np
from nilearn.masking import compute_epi_mask
import tedana.io, tedana.utils

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def restrict_mask(mask, t2star_map, t2_map):
    mask[np.any(np.logical_or(t2star_map == 0, t2_map == 0), axis=t2star_map.ndim - 1)] = 0


def get_tedana_default_mask(ref_img, data):
    LGR.info("Computing EPI mask from first echo")
    first_echo_img = tedana.io.new_nii_like(ref_img, data[:, 0, :])
    mask = compute_epi_mask(first_echo_img)
    RepLGR.info(
        "An initial mask was generated from the first echo using "
        "nilearn's compute_epi_mask function."
    )
    mask_res, _ = make_adaptive_mask(data, mask, getsum=False, threshold=1)
    return mask_res


def make_adaptive_mask(data, mask, getsum=False, threshold=1):
    if getsum == True:
        mask, masksum = tedana.utils.make_adaptive_mask(
            data,
            mask=mask,
            getsum=getsum,
            threshold=threshold,
        )
    else:
        mask = tedana.utils.make_adaptive_mask(
            data,
            mask=mask,
            getsum=getsum,
            threshold=threshold,
        )
        masksum = None
    return mask, masksum
