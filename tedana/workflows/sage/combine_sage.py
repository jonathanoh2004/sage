"""
Functions to optimally combine data across echoes.
"""
import logging
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
        raise ValueError(
            "SAGE requires 5 echos for computing T2 and T2*-weighted optimal combinations"
        )
    if mask.shape != (data.shape[0], 1):
        raise ValueError("Shape of mask must match (data.shape[0], 1)")
    if not (
        t2star_map.shape == s0_I_map.shape
        and s0_I_map.shape == t2_map.shape
        and t2_map.shape == s0_II_map.shape
    ):
        raise ValueError("Shapes of maps must conform to (S x T)")

    w_t2star, w_t2 = weights_sage(tes, t2star_map, s0_I_map, t2_map, s0_II_map)

    assert w_t2star.ndim == w_t2.ndim

    if w_t2star.ndim == 2:
        w_t2star = np.expand_dims(w_t2star, axis=2)
        w_t2 = np.expand_dims(w_t2, axis=2)

    echo_axis = 1

    optcom_t2star = np.sum(w_t2star * (data * np.expand_dims(mask, axis=2)), axis=echo_axis)
    optcom_t2 = np.sum(w_t2 * (data * np.expand_dims(mask, axis=2)), axis=echo_axis)

    return optcom_t2star, optcom_t2


def weights_sage(tes, t2star_map, s0_I_map, t2_map, s0_II_map):
    """
    Computes both T2* and T2-weighted weights for each (voxel, echo)
    pair for gradient echos
    Inputs will either be of shape (S) or (S x T)
    Output will be of shape (S x T) or (S) when fixed in loglin
    """
    if not (
        t2star_map.shape == s0_I_map.shape
        and s0_I_map.shape == t2_map.shape
        and t2_map.shape == s0_II_map.shape
    ):
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

    w_t2star = np.zeros((s0_I_map.shape[0], tes.size, s0_I_map.shape[s0_I_map.ndim - 1]))
    w_t2 = np.zeros((s0_I_map.shape[0], tes.size, s0_I_map.shape[s0_I_map.ndim - 1]))

    w_t2star_I = (s0_I_map * (-1 * tes_indexed_I)) * np.exp(
        (1 / t2star_map) * (-1 * tes_indexed_I)
    )
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

    w_t2star = w_t2star / np.expand_dims(np.sum(w_t2star, axis=echo_axis), axis=echo_axis)
    w_t2 = w_t2 / np.expand_dims(np.sum(w_t2, axis=echo_axis), axis=echo_axis)

    return w_t2star, w_t2
