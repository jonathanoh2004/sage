import os.path

import numpy as np
import pytest
import nilearn

from tedana import combine, io, decay, utils
from tedana.tests.utils_sage import get_test_data_path_sage


@pytest.fixture(scope="module")
def testdata_sage():
    echo_times = np.array([7.9, 27, 58, 77, 96]) / 1000
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
    out_files = [
        os.path.join(get_test_data_path_sage(), name)
        for name in ["T2starmap.nii.gz", "S0Imap.nii.gz", "T2map.nii.gz", "S0IImap.nii.gz"]
    ]

    catd, _ = io.load_data(in_files, n_echos)
    mask = nilearn.image.load_img(in_file_mask).get_fdata().reshape(catd.shape[0], 1)
    exp_t2star, exp_s0_I, exp_t2, exp_s0_II = (
        nilearn.image.load_img(out_file).get_fdata().reshape(catd.shape[0], catd.shape[2])
        for out_file in out_files
    )
    fittypes = ["loglin", "nonlin"]
    fitmodes = ["all", "each"]

    data_dict = {
        "data": catd,
        "echo_times": echo_times,
        "mask": mask,
        "fittypes": fittypes,
        "exp_t2star": exp_t2star,
        "exp_s0_I": exp_s0_I,
        "exp_t2": exp_t2,
        "exp_s0_II": exp_s0_II,
        "fitmodes": fitmodes,
    }
    return data_dict


def test_decay_sage_loglin(testdata_sage):
    catd, tes, mask, fittypes, fitmodes = (
        testdata_sage["data"],
        testdata_sage["echo_times"],
        testdata_sage["mask"],
        testdata_sage["fittypes"],
        testdata_sage["fitmodes"],
    )
    exp_t2star, exp_s0_I, exp_t2, exp_s0_II = (
        testdata_sage["exp_t2star"],
        testdata_sage["exp_s0_I"],
        testdata_sage["exp_t2"],
        testdata_sage["exp_s0_II"],
    )

    t2star, s0_I, t2, delta = decay.fit_decay_sage(catd, tes, mask, fittypes[0], fitmodes[0])
    s0_II = (1 / delta) * s0_I

    np.testing.assert_allclose(t2star, exp_t2star)
    np.testing.assert_allclose(s0_I, exp_s0_I)
    np.testing.assert_allclose(t2, exp_t2)
    np.testing.assert_allclose(s0_II, exp_s0_II)

    t2star, s0_I, t2, delta = decay.fit_decay_sage(catd, tes, mask, fittypes[0], fitmodes[1])
    s0_II = (1 / delta) * s0_I

    np.testing.assert_allclose(t2star, exp_t2star)
    np.testing.assert_allclose(s0_I, exp_s0_I)
    np.testing.assert_allclose(t2, exp_t2)
    np.testing.assert_allclose(s0_II, exp_s0_II)
