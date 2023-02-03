"""
Functions to optimally combine data across echoes.
"""
import logging
import numpy as np
from tedana.workflows.sage import config_sage

LGR = logging.getLogger("GENERAL")


######################################################################################
################################## SAGE OPTCOM #######################################
######################################################################################


def make_optcom_sage(data, tes, t2star_map, s0I_map, t2_map, s0II_map, mask):
    """
    ----- DESCRIPTION -----
        Computes optimal combination combinations of input data based on the
        provided T2* and T2 maps.
    ----- INPUT -----
        data: numpy.ndarray: (S, E, T)
        tes: numpy.ndarray: (E,)
        t2star_map: numpy.ndarray: (S, T)
        s0I_map: numpy.ndarray: (S, T)
        t2_map: numpy.ndarray: (S, T)
        s0II_map: numpy.ndarray: (S, T)
        mask: numpy.ndarray: (S, 1, 1)
    ----- OUTPUT -----
        optcom_t2star: numpy.ndarray: (S, T)
        optcom_t2: numpy.ndarray: (S, T)
        Computes and returns optimal combinations
    """
    axis_echos = config_sage.get_axis_echos()

    w_t2star, w_t2 = weights_sage(data, tes, t2star_map, s0I_map, t2_map, s0II_map)

    optcom_t2star = np.sum(w_t2star * (data * mask), axis=axis_echos)
    optcom_t2 = np.sum(w_t2 * (data * mask), axis=axis_echos)

    return optcom_t2star, optcom_t2


def weights_sage(data, tes, t2star_map, s0I_map, t2_map, s0II_map):
    """
    ----- DESCRIPTION -----
        Computes optimal combination echo weights for each
        (voxel, timepoint) pair as partial derivatives of
        the SAGE decay model.
    ----- INPUT -----
        data: numpy.ndarray(S, E, T)
        tes: numpy.ndarray: (E,)
        t2star_map: numpy.ndarray: (S, T)
        s0I_map: numpy.ndarray: (S, T)
        t2_map: numpy.ndarray: (S, T)
        s0II_map: numpy.ndarray: (S, T)
    ----- OUTPUT -----
        w_t2star: numpy.ndarray: (S, E, T)
        w_t2: numpy.ndarray: (S, E, T)
    """
    n_samps = config_sage.get_n_samps(data)
    n_vols = config_sage.get_n_vols(data)
    axis_echos = config_sage.get_axis_echos()
    tese = tes[-1]
    idx_I = tes < tese / 2
    idx_II = tes >= (tese / 2)
    tes = tes[np.newaxis, :, np.newaxis]

    s0I_map = s0I_map[:, np.newaxis, :]
    t2star_map = t2star_map[:, np.newaxis, :]
    s0II_map = s0II_map[:, np.newaxis, :]
    t2_map = t2_map[:, np.newaxis, :]

    tes_indexed_I = tes[:, idx_I, :]
    tes_indexed_II = tes[:, idx_II, :]
    tese_repeated_II = np.repeat(tese, np.sum(idx_II))[np.newaxis, :, np.newaxis]

    w_t2star = np.zeros((n_samps, tes.size, n_vols))
    w_t2 = np.zeros((n_samps, tes.size, n_vols))

    w_t2star_I = (s0I_map * (-1 * tes_indexed_I)) * np.exp((1 / t2star_map) * (-1 * tes_indexed_I))
    w_t2_I = np.zeros(w_t2star_I.shape)

    const1 = s0II_map * ((-1 * tese_repeated_II) + tes_indexed_II)
    const2 = s0II_map * ((tese_repeated_II - (2 * tes_indexed_II)))
    exp1 = ((1 / t2star_map) - (1 / t2_map)) * (-1 * tese_repeated_II)
    exp2 = ((2 * (1 / t2_map)) - (1 / t2star_map)) * (tes_indexed_II)

    w_t2star_II = const1 * np.exp(exp1 - exp2)
    w_t2_II = const2 * np.exp(exp1 - exp2)

    w_t2star[:, idx_I, :] = w_t2star_I
    w_t2star[:, idx_II, :] = w_t2star_II
    w_t2[:, idx_I, :] = w_t2_I
    w_t2[:, idx_II, :] = w_t2_II

    w_t2star = w_t2star / np.expand_dims(np.sum(w_t2star, axis=axis_echos), axis=axis_echos)
    w_t2 = w_t2 / np.expand_dims(np.sum(w_t2, axis=axis_echos), axis=axis_echos)

    return w_t2star, w_t2
