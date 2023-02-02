import logging
import numpy as np
from nilearn.masking import compute_epi_mask
import tedana.io, tedana.utils
from tedana.workflows.sage import config_sage

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


def get_adaptive_mask_clf(mask, masksum, data, cmdline_args):
    if cmdline_args.mask_type in ["tedana", "tedana_adaptive"]:
        n_samps = config_sage.get_n_samps(data)
        # create an adaptive mask with at least 3 good echoes, for classification
        mask_clf, masksum_clf = make_adaptive_mask(
            data,
            mask,
            getsum=config_sage.get_getsum_masksum_clf(cmdline_args.mask_type),
            threshold=config_sage.get_threshold_masksum_clf(),
        )
        if masksum_clf is None:
            masksum_clf = masksum

        LGR.debug("Retaining {}/{} samples for denoising".format(mask.sum(), n_samps))
        RepLGR.info(
            "A two-stage masking procedure was applied, in which a liberal mask "
            "(including voxels with good data in at least the first echo) was used for "
            "optimal combination, T2*/T2/S0I/S0II estimation, and denoising, while a more "
            "conservative mask (restricted to voxels with good data in at least the first "
            "three echoes) was used for the component classification procedure."
        )
        LGR.debug("Retaining {}/{} samples for classification".format(mask_clf.sum(), n_samps))
    else:
        mask_clf, masksum_clf = mask, masksum
    return mask_clf, masksum_clf
