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


def unmask(dict_arrs, mask):
    """
    Convenience function to unmask a list of arrays
    and check for None values.
    ----- INPUT -----
        arrs: dict[numpy.ndarray | None]
    ----- OUTPUT -----
        res: dict[numpy.ndarray, ...]
    """
    res = {}
    for key, arr in dict_arrs.items():
        if arr is not None:
            res[key] = tedana.utils.unmask(arr, mask)
    return res


def apply_t2s_floor(t2star, t2, echo_times):
    """
    Adaptation of _apply_t2s_floor in tedana.decay module.
    """

    eps = np.finfo(dtype=t2star.dtype).eps  # smallest value for datatype
    # Exclude voxels where t2star or t2 is 0 at some timepoint
    nonzerovox = np.all(np.logical_or(t2star != 0, t2 != 0), axis=1)

    tese = echo_times[-1]
    temp_arr = np.zeros((echo_times.size, t2star.shape[0], t2star.shape[1]))
    temp_arr = temp_arr[:, nonzerovox, :]
    # t2star = np.reshape(t2star, (1, t2star.shape[0], t2star.shape[1]))
    # t2 = np.reshape(t2, (1, t2.shape[0], t2.shape[1]))
    temp_arr[(echo_times < tese / 2), :, :] = np.exp(
        -1
        * echo_times[echo_times < tese / 2][:, np.newaxis, np.newaxis]
        * (1 / t2star[np.newaxis, nonzerovox, :])
    )
    temp_arr[(echo_times > tese / 2), :, :] = np.exp(
        -1 * tese * ((1 / t2star[np.newaxis, nonzerovox, :]) - (1 / t2[np.newaxis, nonzerovox, :]))
    ) * np.exp(
        -1
        * echo_times[echo_times > tese / 2][:, np.newaxis, np.newaxis]
        * ((2 * (1 / t2[np.newaxis, nonzerovox, :])) - (1 / t2star[np.newaxis, nonzerovox, :]))
    )
    bad_voxel_idx = np.any(temp_arr == 0, axis=(0, 2))

    t2star_corrected = t2star.copy()
    t2_corrected = t2.copy()
    t2star_corrected[bad_voxel_idx, :] = np.min(-echo_times) / np.log(eps)
    t2_corrected[bad_voxel_idx, :] = np.min(-echo_times) / np.log(eps)
    return t2star_corrected, t2_corrected


def setup_loggers(repname, quiet, debug):
    tedana.utils.setup_loggers(logname=None, repname=repname, quiet=quiet, debug=debug)


def teardown_loggers():
    tedana.utils.teardown_loggers()
