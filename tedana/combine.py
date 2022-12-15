"""
Functions to optimally combine data across echoes.
"""
import logging
import copy

import numpy as np

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


######################################################################################
################################## SAGE OPTCOM #######################################
######################################################################################


def make_optcom_sage(data, tes, t2star_map, s0_I_map, t2_map, s0_II_map, mask):
    if data.ndim != 3:
        raise ValueError("Data should be of dimension (S x E x T)")
    if data.shape[1] != len(tes):
        raise ValueError(
            "Second dimension of data ({0}) does not match number "
            "of echoes provided (tes; {1})".format(data.shape[1], len(tes))
        )
    if len(tes) != 5:
        raise ValueError("SAGE requires 5 echos for computing T2 and T2*-weighted optimal combinations")
    if mask.shape != (data.shape[0], 1):
        raise ValueError("Shape of mask must match (data.shape[0], 1)")
    if not (t2star_map.shape == s0_I_map.shape and s0_I_map.shape == t2_map.shape and t2_map.shape == s0_II_map.shape):
        raise ValueError("Shapes of maps must conform to (S x T)")

    # w_t2star_I, w_t2_I = weights_sage_I(tes, t2star_map, s0_I_map)
    # w_t2star_II, w_t2_II = weights_sage_II(tes, t2star_map, t2_map, s0_II_map)

    w_t2star, w_t2 = weights_sage(tes, t2star_map, s0_I_map, t2_map, s0_II_map)

    assert (w_t2star.ndim == w_t2.ndim)

    if w_t2star.ndim == 2:
        w_t2star = np.expand_dims(w_t2star, axis=2)
        w_t2 = np.expand_dims(w_t2, axis=2)

    echo_axis = 1

    optcom_t2star = np.sum(w_t2star * (data * np.expand_dims(mask.astype(bool), axis=2)), axis=echo_axis)
    optcom_t2 = np.sum(w_t2 * (data * np.expand_dims(mask.astype(bool), axis=2)), axis=echo_axis)

    return optcom_t2star, optcom_t2


def weights_sage(tes, t2star_map, s0_I_map, t2_map, s0_II_map):
    '''
    Computes both T2* and T2-weighted weights for each (voxel, echo)
    pair for gradient echos
    Output will either be of shape (V x E) or (V x E x T)
    '''
    if not (t2star_map.shape == s0_I_map.shape and s0_I_map.shape == t2_map.shape and t2_map.shape == s0_II_map.shape):
        raise ValueError("maps must be of same shape")
    
    maps_ndim = s0_I_map.ndim
    tese = tes[-1]
    idx_I = tes < tese / 2
    idx_II = tes >= (tese / 2)
    tes = tes[np.newaxis, :, np.newaxis]
    
    echo_axis = 1

    if maps_ndim == 1:
        s0_I_map = s0_I_map[:, np.newaxis, np.newaxis]
        t2star_map = t2star_map[:, np.newaxis, np.newaxis]
        s0_II_map = s0_II_map[:, np.newaxis, np.newaxis]
        t2_map = t2_map[:, np.newaxis, np.newaxis]
    elif maps_ndim == 2:
        s0_I_map = s0_I_map[:, :, np.newaxis].swapaxes(1, 2)
        t2star_map = t2star_map[:, :, np.newaxis].swapaxes(1, 2)
        s0_II_map = s0_II_map[:, :, np.newaxis].swapaxes(1, 2)
        t2_map = t2_map[:, :, np.newaxis].swapaxes(1, 2)
    else:
        raise ValueError("Maps are of invalid shape")

    tes_indexed_I = tes[:, idx_I, :]
    tes_indexed_II = tes[:, idx_II, :]
    tese_repeated_II = np.repeat(tese, np.sum(idx_II))[np.newaxis, :, np.newaxis]

    # if maps_ndim == 1:
    #     tes_indexed_I = tes[:, idx_I]
    #     tes_indexed_II = tes[:, idx_II]
    #     tese_repeated_II = np.repeat(tese, np.sum(idx_II))[np.newaxis, :]
    # elif maps_ndim == 2:
    #     tes = np.expand_dims(tes, axis=2)
    #     tes_indexed_I = tes[:, idx_I, :]
    #     tes_indexed_II = tes[:, idx_II, :]
    #     tese_repeated_II = np.repeat(tese, np.sum(idx_II))[np.newaxis, :, np.newaxis]
    # else:
    #     raise ValueError("Maps are of invalid shape")
    
    w_t2star = np.zeros((s0_I_map.shape[0], tes.size, s0_I_map.shape[s0_I_map.ndim - 1]))
    w_t2 = np.zeros((s0_I_map.shape[0], tes.size, s0_I_map.shape[s0_I_map.ndim - 1]))
    
    w_t2star_I = (s0_I_map * (-1 * tes_indexed_I)) * np.exp((1 / t2star_map) * (-1 * tes_indexed_I))
    w_t2_I = np.zeros(w_t2star_I.shape)

    const1 = s0_II_map * ((-1 * tese_repeated_II) + tes_indexed_II)
    const2 = s0_II_map * ((tese_repeated_II - (2 * tes_indexed_II)))
    exp1 = ((1 / t2star_map) - (1 / t2_map)) * (-1 * tese_repeated_II)
    exp2 = ((2 * (1 / t2_map)) - (1 / t2star_map)) * (tes_indexed_II)

    w_t2star_II = const1 * np.exp(exp1 - exp2)
    w_t2_II = const2 * np.exp(exp1 - exp2)

    w_t2star[:, idx_I, :] = w_t2star_I
    w_t2star[:, idx_II, :] = w_t2star_II
    w_t2[:, idx_I, :] = w_t2_I
    w_t2[:, idx_II, :] = w_t2_II

    assert (w_t2star.shape == w_t2.shape)

    # If all values across echos are 0, set to 1 to avoid
    # divide-by-zero errors
    # w_t2star_II[np.where(np.all(w_t2star_II == 0, axis=echo_axis)), :] = 1
    # w_t2_II[np.where(np.all(w_t2_II == 0, axis=echo_axis)), :] = 1
    # w_t2star_I[np.where(np.all(w_t2star_I == 0, axis=echo_axis)), :] = 1

    # normalize
    # w_t2.mean() was -1.945989
    # w_t2star.mean() was -1.6607357
    w_t2star = w_t2star / np.expand_dims(np.sum(w_t2star, axis=echo_axis), axis=echo_axis)
    w_t2 = w_t2 / np.expand_dims(np.sum(w_t2, axis=echo_axis), axis=echo_axis)
    # w_t2star_II = w_t2star_II / np.expand_dims(np.sum(w_t2star_II, axis=echo_axis), axis=echo_axis)
    # w_t2_II = w_t2_II / np.expand_dims(np.sum(w_t2_II, axis=echo_axis), axis=echo_axis)

    return w_t2star, w_t2


def weights_sage_I(tes, t2star_map, s0_I_map):
    '''
    Computes both T2* and T2-weighted weights for each (voxel, echo)
    pair for gradient echos
    Output will either be of shape (V x E*) or (V x E* x T)
    Where E* represents only the relevant echos
    '''
    if not (s0_I_map.shape == t2star_map.shape):
        raise ValueError("maps must be of same shape")
    else:
        maps_ndim = s0_I_map.ndim

    tese = tes[-1]
    idx_I = tes < tese / 2
    tes = np.expand_dims(tes, axis=0)

    s0_I_map = np.expand_dims(s0_I_map, axis=1)
    t2star_map = np.expand_dims(t2star_map, axis=1)

    if maps_ndim == 1:
        tes_indexed = tes[:, idx_I]
    elif maps_ndim == 2:
        tes = np.expand_dims(tes, axis=2)
        tes_indexed = tes[:, idx_I, :]
    
    w_I = (s0_I_map * (-1 * tes_indexed)) * np.exp((1 / t2star_map) * (-1 * tes_indexed))

    echo_axis = 1

    # If all values across echos are 0, set to 1 to avoid
    # divide-by-zero errors
    # w_I[np.where(np.all(w_I == 0, axis=echo_axis)), :] = 1

    # normalize
    w_I = w_I / np.expand_dims(np.sum(w_I, axis=echo_axis), axis=echo_axis)

    # combined_t2star_I = np.zeros((data.shape[0], data.shape[2]))

    # for samp_idx in range(data.shape[0]):
    #     combined_t2star_I[samp_idx, :] = np.average(
    #         data[samp_idx, idx_I[0, :], :], axis=0, weights=w_I[samp_idx, :]
    #     )

    # # derivative with respect to t2 is 0, so corresponding weight is always all zeros
    # combined_t2_I = np.tile([0], combined_t2star_I.shape)

    return w_I, np.zeros(w_I.shape)


def weights_sage_II(tes, t2star_map, t2_map, s0_II_map):
    '''
    Computes both T2* and T2-weighted weights for each (voxel, echo)
    pair for asymmetric and spin echos
    Output will either be of shape (V x E*) or (V x T x E*)
    Where E* represents only the relevant echos
    '''
    if not (s0_II_map.shape == t2star_map.shape and t2star_map.shape == t2_map.shape):
        raise ValueError("maps must be of same shape")
    else:
        maps_ndim = s0_II_map.ndim

    tese = tes[-1]
    idx_II = tes >= (tese / 2)
    tes = np.expand_dims(tes, axis=0)

    s0_II_map = np.expand_dims(s0_II_map, axis=1)
    t2star_map = np.expand_dims(t2star_map, axis=1)
    t2_map = np.expand_dims(t2_map, axis=1)

    if maps_ndim == 1:
        tes_indexed = tes[:, idx_II]
    elif maps_ndim == 2:
        tes = np.expand_dims(tes, axis=2)
        tes_indexed = tes[:, idx_II, :]

    tese_repeated = np.repeat(tese, np.sum(idx_II))[np.newaxis, :, np.newaxis]

    const1 = s0_II_map * ((-1 * tese) + tes_indexed)
    const2 = s0_II_map * ((tese - (2 * tes_indexed)))
    exp1 = ((1 / t2star_map) - (1 / t2_map)) * (-1 * tese_repeated)
    exp2 = ((2 * (1 / t2_map)) - (1 / t2star_map)) * (tes_indexed)

    w_t2star_II = const1 * np.exp(exp1 - exp2)
    w_t2_II = const2 * np.exp(exp1 - exp2)

    assert (w_t2star_II.shape == w_t2_II.shape)
    echo_axis = 1

    # If all values across echos are 0, set to 1 to avoid
    # divide-by-zero errors
    # w_t2star_II[np.where(np.all(w_t2star_II == 0, axis=echo_axis)), :] = 1
    # w_t2_II[np.where(np.all(w_t2_II == 0, axis=echo_axis)), :] = 1

    # normalize
    w_t2star_II = w_t2star_II / np.expand_dims(np.sum(w_t2star_II, axis=echo_axis), axis=echo_axis)
    w_t2_II = w_t2_II / np.expand_dims(np.sum(w_t2_II, axis=echo_axis), axis=echo_axis)

    # combined_t2star_II = np.zeros((data.shape[0], data.shape[2]))
    # combined_t2_II = np.zeros((data.shape[0], data.shape[2]))

    # for samp_idx in range(data.shape[0]):
    #     combined_t2star_II[samp_idx, :] = np.average(
    #         data[samp_idx, idx_II[0, :], :], axis=0, weights=w_t2star_II[samp_idx, :]
    #     )
    #     combined_t2_II[samp_idx, :] = np.average(
    #         data[samp_idx, idx_II[0, :], :], axis=0, weights=w_t2_II[samp_idx, :]
    #     )

    return w_t2star_II, w_t2_II


######################################################################################
################################## ORIGINAL OPTCOM ###################################
######################################################################################


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
