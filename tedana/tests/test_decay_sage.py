import os.path

import numpy as np
import pytest
import nilearn

from tedana import combine_sage, decay_sage, io, utils
from tedana.tests.utils_sage import get_test_data_path_sage


@pytest.fixture(scope="module")
def input_sage():
    echo_times = np.array([7.9, 27.0, 58.0, 77.0, 96.0]) / 1000
    n_echos = len(echo_times)
    in_files = [
        os.path.join(
            get_test_data_path_sage(), "Multigre_SAGE_e{0}_tshift_bet.nii.gz".format(i + 1)
        )
        for i in range(n_echos)
    ]
    in_file_mask = os.path.join(
        get_test_data_path_sage(), "Multigre_SAGE_e2_tshift_tmean_bet_mask.nii.gz"
    )

    catd, _ = io.load_data(in_files, n_echos)
    mask = nilearn.image.load_img(in_file_mask).get_fdata().reshape(catd.shape[0], 1)

    fittypes = ["loglin", "nonlin"]
    fitmodes = ["all", "each"]

    data_dict = {
        "data": catd,
        "echo_times": echo_times,
        "mask": mask,
        "fittypes": fittypes,
        "fitmodes": fitmodes,
    }
    return data_dict


@pytest.fixture(scope="module")
def exp_output_loglin():
    out_dir = "loglin"
    out_files = [
        os.path.join(get_test_data_path_sage(), out_dir, name)
        for name in [
            "T2starmap_Loglin.nii.gz",
            "S0Imap_Loglin.nii.gz",
            "T2map_Loglin.nii.gz",
            "S0IImap_Loglin.nii.gz",
        ]
    ]
    exp_t2star, exp_s0_I, exp_t2, exp_s0_II = (
        nilearn.image.load_img(out_file).get_fdata() for out_file in out_files
    )

    data_dict = {
        "exp_t2star": exp_t2star,
        "exp_s0_I": exp_s0_I,
        "exp_t2": exp_t2,
        "exp_s0_II": exp_s0_II,
    }
    return data_dict


@pytest.fixture(scope="module")
def exp_output_nonlin():
    out_dir = "nonlin"
    out_files = [
        os.path.join(get_test_data_path_sage(), out_dir, name)
        for name in ["RMSPE_NLSQ.nii.gz", "S0_NLSQ.nii.gz", "T2_NLSQ.nii.gz", "T2s_NLSQ.nii.gz"]
    ]
    exp_rmspe, exp_s0, exp_t2, exp_t2star = (
        nilearn.image.load_img(out_file).get_fdata() for out_file in out_files
    )

    data_dict = {
        "exp_rmspe": exp_rmspe,
        "exp_s0": exp_s0,
        "exp_t2": exp_t2,
        "exp_t2star": exp_t2star,
    }
    return data_dict


def test_decay_sage_loglin(input_sage, exp_output_loglin):
    catd, tes, mask, fittypes, fitmodes = (
        input_sage["data"],
        input_sage["echo_times"],
        input_sage["mask"],
        input_sage["fittypes"],
        input_sage["fitmodes"],
    )
    exp_t2star, exp_s0_I, exp_t2, exp_s0_II = (
        exp_output_loglin["exp_t2star"],
        exp_output_loglin["exp_s0_I"],
        exp_output_loglin["exp_t2"],
        exp_output_loglin["exp_s0_II"],
    )

    exp_t2star = exp_t2star.reshape(catd.shape[0], catd.shape[2])
    exp_s0_I = exp_s0_I.reshape(catd.shape[0], catd.shape[2])
    exp_t2 = exp_t2.reshape(catd.shape[0], catd.shape[2])
    exp_s0_II = exp_s0_II.reshape(catd.shape[0], catd.shape[2])

    t2star, s0_I, t2, delta, _ = decay_sage.fit_decay_sage(catd, tes, mask, "loglin", "all")
    s0_II = (1 / delta) * s0_I

    np.testing.assert_allclose(t2star, exp_t2star)
    np.testing.assert_allclose(s0_I, exp_s0_I)
    np.testing.assert_allclose(t2, exp_t2)
    np.testing.assert_allclose(s0_II, exp_s0_II)

    # t2star, s0_I, t2, delta, _ = decay.fit_decay_sage(catd, tes, mask, fittypes[0], fitmodes[1])
    # s0_II = (1 / delta) * s0_I

    # np.testing.assert_allclose(t2star, exp_t2star)
    # np.testing.assert_allclose(s0_I, exp_s0_I)
    # np.testing.assert_allclose(t2, exp_t2)
    # np.testing.assert_allclose(s0_II, exp_s0_II)


def test_decay_sage_nonlin(input_sage, exp_output_nonlin):
    catd, tes, mask, fittypes, fitmodes = (
        input_sage["data"],
        input_sage["echo_times"],
        input_sage["mask"],
        input_sage["fittypes"],
        input_sage["fitmodes"],
    )

    exp_rmspe, exp_s0, exp_t2, exp_t2star = (
        exp_output_nonlin["exp_rmspe"],
        exp_output_nonlin["exp_s0"],
        exp_output_nonlin["exp_t2"],
        exp_output_nonlin["exp_t2star"],
    )

    exp_rmspe = exp_rmspe.reshape(catd.shape[0], catd.shape[2])
    exp_s0 = exp_s0.reshape(catd.shape[0], catd.shape[2])
    exp_t2 = exp_t2.reshape(catd.shape[0], catd.shape[2])
    exp_t2star = exp_t2star.reshape(catd.shape[0], catd.shape[2])

    t2star, s0_I, t2, delta, rmspe = decay_sage.fit_decay_sage(catd, tes, mask, "nonlin", "each")
    # s0_II = (1 / delta) * s0_I

    np.testing.assert_allclose(rmspe, exp_rmspe)
    np.testing.assert_allclose(s0_I, exp_s0)
    np.testing.assert_allclose(t2, exp_t2)
    np.testing.assert_allclose(t2star, exp_t2star)

    # t2star, s0_I, t2, delta, _ = decay.fit_decay_sage(catd, tes, mask, fittypes[0], fitmodes[1])
    # s0_II = (1 / delta) * s0_I

    # np.testing.assert_allclose(t2star, exp_t2star)
    # np.testing.assert_allclose(s0_I, exp_s0_I)
    # np.testing.assert_allclose(t2, exp_t2)
    # np.testing.assert_allclose(s0_II, exp_s0_II)
