"""
Functions to optimally combine data across echoes.
"""
import logging
import copy

import numpy as np

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _combine_t2s(data, tes, ft2s, report=True):
    """
    Combine data across echoes using weighted averaging according to voxel-
    (and sometimes volume-) wise estimates of T2*.

    This method was proposed in :footcite:t:`posse1999enhancement`.

    Parameters
    ----------
    data : (M x E x T) array_like
        Masked data.
    tes : (1 x E) array_like
        Echo times in milliseconds.
    ft2s : (M [x T] X 1) array_like
        Either voxel-wise or voxel- and volume-wise estimates of T2*.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (M x T) :obj:`numpy.ndarray`
        Data combined across echoes according to T2* estimates.

    References
    ----------
    .. footbibliography::
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "T2* combination method \\citep{posse1999enhancement}."
        )

    n_vols = data.shape[-1]
    alpha = tes * np.exp(-tes / ft2s)
    if alpha.ndim == 2:
        # Voxel-wise T2 estimates
        alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    elif alpha.ndim == 3:
        # Voxel- and volume-wise T2 estimates
        # alpha is currently (S, T, E) but should be (S, E, T) like mdata
        alpha = np.swapaxes(alpha, 1, 2)

        # If all values across echos are 0, set to 1 to avoid
        # divide-by-zero errors
        ax0_idx, ax2_idx = np.where(np.all(alpha == 0, axis=1))
        alpha[ax0_idx, :, ax2_idx] = 1.0
    combined = np.average(data, axis=1, weights=alpha)
    return combined


def _combine_paid(data, tes, report=True):
    """
    Combine data across echoes using SNR/signal and TE via the
    parallel-acquired inhomogeneity desensitized (PAID) ME-fMRI combination
    method.

    This method was first proposed in :footcite:t:`poser2006bold`.

    Parameters
    ----------
    data : (M x E x T) array_like
        Masked data.
    tes : (1 x E) array_like
        Echo times in milliseconds.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (M x T) :obj:`numpy.ndarray`
        Data combined across echoes according to SNR/signal.

    References
    ----------
    .. footbibliography::
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "parallel-acquired inhomogeneity desensitized (PAID) "
            "combination method \\citep{poser2006bold}."
        )

    n_vols = data.shape[-1]
    snr = data.mean(axis=-1) / data.std(axis=-1)
    alpha = snr * tes
    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    combined = np.average(data, axis=1, weights=alpha)
    return combined


def make_optcom(data, tes, adaptive_mask, t2s=None, combmode="t2s", verbose=True):
    """
    Optimally combine BOLD data across TEs, using only those echos with reliable signal
    across at least three echos. If the number of echos providing reliable signal is greater
    than three but less than the total number of collected echos, we assume that later
    echos do not provided meaningful signal.

    Parameters
    ----------
    data : (S x E x T) :obj:`numpy.ndarray`
        Concatenated BOLD data.
    tes : (E,) :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    t2s : (S [x T]) :obj:`numpy.ndarray` or None, optional
        Estimated T2* values. Only required if combmode = 't2s'.
        Default is None.
    combmode : {'t2s', 'paid'}, optional
        How to combine data. Either 'paid' or 't2s'. If 'paid', argument 't2s'
        is not required. Default is 't2s'.
    verbose : :obj:`bool`, optional
        Whether to print status updates. Default is True.

    Returns
    -------
    combined : (S x T) :obj:`numpy.ndarray`
        Optimally combined data.

    Notes
    -----
    This function supports both the ``'t2s'`` method :footcite:p:`posse1999enhancement`
    and the ``'paid'`` method :footcite:p:`poser2006bold`.
    The ``'t2s'`` method operates according to the following logic:

    1.  Estimate voxel- and TE-specific weights based on estimated :math:`T_2^*`:

            .. math::
                w(T_2^*)_n = \\frac{TE_n * exp(\\frac{-TE}\
                {T_{2(est)}^*})}{\\sum TE_n * exp(\\frac{-TE}{T_{2(est)}^*})}
    2.  Perform weighted average per voxel and TR across TEs based on weights
        estimated in the previous step.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3D (S x E x T)")

    if len(tes) != data.shape[1]:
        raise ValueError(
            "Number of echos provided does not match second "
            "dimension of input data: {0} != "
            "{1}".format(len(tes), data.shape[1])
        )

    if adaptive_mask.ndim != 1:
        raise ValueError("Mask is not 1D")
    elif adaptive_mask.shape[0] != data.shape[0]:
        raise ValueError(
            "Mask and data do not have same number of "
            "voxels/samples: {0} != {1}".format(adaptive_mask.shape[0], data.shape[0])
        )

    if combmode not in ["t2s", "paid"]:
        raise ValueError("Argument 'combmode' must be either 't2s' or 'paid'")
    elif combmode == "t2s" and t2s is None:
        raise ValueError("Argument 't2s' must be supplied if 'combmode' is set to 't2s'.")
    elif combmode == "paid" and t2s is not None:
        LGR.warning(
            "Argument 't2s' is not required if 'combmode' is 'paid'. "
            "'t2s' array will not be used."
        )

    if combmode == "paid":
        LGR.info(
            "Optimally combining data with parallel-acquired "
            "inhomogeneity desensitized (PAID) method"
        )
    else:
        if t2s.ndim == 1:
            msg = "Optimally combining data with voxel-wise T2* estimates"
        else:
            msg = "Optimally combining data with voxel- and volume-wise T2* estimates"
        LGR.info(msg)

    echos_to_run = np.unique(adaptive_mask)
    # When there is one good echo, use two
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    tes = np.array(tes)[np.newaxis, ...]  # (1 x E) array_like
    combined = np.zeros((data.shape[0], data.shape[2]))
    report = True
    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            # Use the first two echoes for cases where there are
            # either one or two good echoes
            voxel_idx = np.where(np.logical_and(adaptive_mask > 0, adaptive_mask <= echo_num))[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        if combmode == "paid":
            combined[voxel_idx, :] = _combine_paid(
                data[voxel_idx, :echo_num, :], tes[:, :echo_num]
            )
        else:
            t2s_ = t2s[..., np.newaxis]  # add singleton

            combined[voxel_idx, :] = _combine_t2s(
                data[voxel_idx, :echo_num, :],
                tes[:, :echo_num],
                t2s_[voxel_idx, ...],
                report=report,
            )
        report = False

    return combined


def make_optcom_sage(data, tes, t2star_map, s0_I_map, t2_map, s0_II_map, mask):

    if data.ndim != 3:
        raise ValueError("Input data must be 3D (S x E x T)")

    if tes.shape[0] != data.shape[1]:
        raise ValueError(
            "Number of echos provided does not match second "
            "dimension of input data: {0} != "
            "{1}".format(len(tes), data.shape[1])
        )

    data = data[mask]

    alpha_t2star_I, alpha_t2_I = weights_sage_I(tes, t2star_map, s0_I_map)
    alpha_t2star_II, alpha_t2_II = weights_sage_II(tes, t2star_map, t2_map, s0_II_map)

    alpha_t2star_I = np.expand_dims(alpha_t2star_I, axis=2)
    alpha_t2star_II = np.expand_dims(alpha_t2star_II, axis=2)
    alpha_t2_I = np.expand_dims(alpha_t2_I, axis=2)
    alpha_t2_II = np.expand_dims(alpha_t2_II, axis=2)

    idx_I = tes < (tes[-1] / 2)
    idx_II = tes >= (tes[-1] / 2)

    com1 = copy.deepcopy(data)
    com2 = copy.deepcopy(data)

    com1[:, idx_I, :] = (alpha_t2star_I / 2) * data[:, idx_I, :]
    com1[:, idx_II, :] = (alpha_t2star_II / 2) * data[:, idx_II, :]

    com2[:, idx_I, :] = (alpha_t2_I / 2) * data[:, idx_I, :]
    com2[:, idx_II, :] = (alpha_t2_II / 2) * data[:, idx_II, :]

    optcom_t2star = np.sum(com1, axis=1)
    optcom_t2 = np.sum(com2, axis=1)

    return optcom_t2star, optcom_t2


def weights_sage_I(tes, t2star_map, s0_I_map):
    tese = tes[-1]
    idx_I = tes < tese / 2
    tes = np.expand_dims(tes, axis=0)
    s0_I_map = np.expand_dims(s0_I_map, axis=1)
    t2star_map = np.expand_dims(t2star_map, axis=1)

    alpha_I = (s0_I_map * (-1 * tes[0, idx_I])) * np.exp((1 / t2star_map) * (-1 * tes[0, idx_I]))

    # If all values across echos are 0, set to 1 to avoid
    # divide-by-zero errors
    alpha_I[np.where(np.all(alpha_I == 0, axis=1)), :] = 1

    # normalize
    alpha_I = alpha_I / np.expand_dims(np.sum(alpha_I, axis=1), axis=1)

    # combined_t2star_I = np.zeros((data.shape[0], data.shape[2]))

    # for samp_idx in range(data.shape[0]):
    #     combined_t2star_I[samp_idx, :] = np.average(
    #         data[samp_idx, idx_I[0, :], :], axis=0, weights=alpha_I[samp_idx, :]
    #     )

    # # derivative with respect to t2 is 0, so corresponding weight is always all zeros
    # combined_t2_I = np.tile([0], combined_t2star_I.shape)

    return alpha_I, np.zeros(alpha_I.shape)


def weights_sage_II(tes, t2star_map, t2_map, s0_II_map):
    tese = tes[-1]
    idx_II = tes >= (tese / 2)

    tes = np.expand_dims(tes, axis=0)
    s0_II_map = np.expand_dims(s0_II_map, axis=1)
    t2_map = np.expand_dims(t2_map, axis=1)
    t2star_map = np.expand_dims(t2star_map, axis=1)

    const1 = s0_II_map * ((-1 * tese) + tes[0, idx_II])
    const2 = s0_II_map * ((tese - (2 * tes[0, idx_II])))
    exp1 = ((1 / t2star_map) - (1 / t2_map)) * (-1 * tese)
    exp2 = ((2 * (1 / t2_map)) - (1 / t2star_map)) * (tes[0, idx_II])

    alpha_t2star_II = const1 * np.exp(exp1 - exp2)
    alpha_t2_II = const2 * np.exp(exp1 - exp2)

    # If all values across echos are 0, set to 1 to avoid
    # divide-by-zero errors
    alpha_t2star_II[np.where(np.all(alpha_t2star_II == 0, axis=1)), :] = 1
    alpha_t2_II[np.where(np.all(alpha_t2_II == 0, axis=1)), :] = 1

    # normalize
    alpha_t2star_II = alpha_t2star_II / np.expand_dims(np.sum(alpha_t2star_II, axis=1), axis=1)
    alpha_t2_II = alpha_t2_II / np.expand_dims(np.sum(alpha_t2_II, axis=1), axis=1)

    # combined_t2star_II = np.zeros((data.shape[0], data.shape[2]))
    # combined_t2_II = np.zeros((data.shape[0], data.shape[2]))

    # for samp_idx in range(data.shape[0]):
    #     combined_t2star_II[samp_idx, :] = np.average(
    #         data[samp_idx, idx_II[0, :], :], axis=0, weights=alpha_t2star_II[samp_idx, :]
    #     )
    #     combined_t2_II[samp_idx, :] = np.average(
    #         data[samp_idx, idx_II[0, :], :], axis=0, weights=alpha_t2_II[samp_idx, :]
    #     )

    return alpha_t2star_II, alpha_t2_II
