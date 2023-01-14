import numpy as np


def get_ind_vars(tes):
    tese = tes[-1]
    x_s0_I = np.ones(tes.size)
    x_delta = np.array([0, 0, -1, -1, -1])
    x_r2star = np.array(
        [
            -1 * tes[0],
            -1 * tes[1],
            tes[2] - tese,
            tes[3] - tese,
            0,
        ]
    )
    x_r2 = np.array([0, 0, tese - (2 * tes[2]), tese - (2 * tes[3]), -1 * tese])

    X = np.column_stack([x_s0_I, x_delta, x_r2star, x_r2])
    return X


def get_dep_vars(data, mask):
    n_samps, n_echos, n_vols = data.shape
    Y = data.swapaxes(1, 2).reshape((n_samps * n_vols, -1)).T
    Y = np.log(Y) * (np.repeat(mask, axis=0, repeats=n_vols).T)
    Y[~np.isfinite(Y)] = 0
    return Y


def _reshape_arrs(arrs=None, shape=None):
    if arrs is None or shape is None:
        return None
    for arr in arrs:
        arr = arr.reshape(shape)
    return arrs


def fit_loglinear_sage(data, tes, mask):
    n_samps, n_echos, n_vols = data.shape

    X = get_ind_vars(tes)
    Y = get_dep_vars(data, mask)

    betas = np.linalg.lstsq(X, Y, rcond=None)[0]

    betas[~np.isfinite(betas)] = 0

    s0_I_map = np.exp(betas[0, :]).T
    delta_map = np.exp(betas[1, :]).T
    s0_II_map = s0_I_map / delta_map
    t2star_map = 1 / betas[2, :].T
    t2_map = 1 / betas[3, :].T

    if n_vols > 1:
        s0_I_map = s0_I_map.reshape(n_samps, n_vols)
        s0_II_map = s0_II_map.reshape(n_samps, n_vols)
        delta_map = delta_map.reshape(n_samps, n_vols)
        t2star_map = t2star_map.reshape(n_samps, n_vols)
        t2_map = t2_map.reshape(n_samps, n_vols)

    return t2star_map, s0_I_map, t2_map, s0_II_map, delta_map, None
