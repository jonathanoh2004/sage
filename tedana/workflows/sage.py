"""
Estimate T2, T2*, S0_I, and S0_II for SAGE ME-fMRI according to (cite paper)
and combine data across TEs according to (cite paper)
"""


import argparse
import logging
import os
import os.path as op

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from tedana import __version__, combine, decay, io, utils
from tedana.workflows.parser_utils import is_valid_file


def _get_parser():
    """
    Parses command line inputs for tedana

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser()
    # Argument parser follow templtate provided by RalphyZ
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-d",
        dest="data",
        nargs="+",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Multi-echo dataset for analysis. May be a "
            "single file with spatially concatenated data "
            "or a set of echo-specific files, in the same "
            "order as the TEs are listed in the -e "
            "argument."
        ),
        required=True,
    )
    required.add_argument(
        "-e",
        dest="tes",
        nargs="+",
        metavar="TE",
        type=float,
        help="Echo times (in ms). E.g., 15.0 39.0 63.0",
        required=True,
    )
    optional.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    optional.add_argument(
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Binary mask of voxels to include in TE "
            "Dependent ANAlysis. Must be in the same "
            "space as `data`."
        ),
        default=None,
    )
    optional.add_argument(
        "--prefix", dest="prefix", type=str, help="Prefix for filenames generated.", default=""
    )
    optional.add_argument(
        "--convention",
        dest="convention",
        action="store",
        choices=["orig", "bids"],
        help=("Filenaming convention. bids will use the latest BIDS derivatives version."),
        default="bids",
    )
    optional.add_argument(
        "--fittype",
        dest="fittype",
        action="store",
        choices=["loglin", "curvefit"],
        help="Desired Fitting Method"
        '"loglin" means that a linear model is fit'
        " to the log of the data, default"
        '"curvefit" means that a more computationally'
        "demanding monoexponential model is fit"
        "to the raw data",
        default="loglin",
    )
    optional.add_argument(
        "--fitmode",
        dest="fitmode",
        action="store",
        choices=["all", "ts"],
        help=(
            "Monoexponential model fitting scheme. "
            '"all" means that the model is fit, per voxel, '
            "across all timepoints. "
            '"ts" means that the model is fit, per voxel '
            "and per timepoint."
        ),
        default="all",
    )
    optional.add_argument(
        "--combmode",
        dest="combmode",
        action="store",
        choices=["t2s", "paid"],
        help=("Combination scheme for TEs: t2s (Posse 1999, default), paid (Poser)"),
        default="t2s",
    )
    optional.add_argument(
        "--n-threads",
        dest="n_threads",
        type=int,
        action="store",
        help=(
            "Number of threads to use. Used by "
            "threadpoolctl to set the parameter outside "
            "of the workflow function. Higher numbers of "
            "threads tend to slow down performance on "
            "typical datasets. Default is 1."
        ),
        default=1,
    )
    optional.add_argument(
        "--debug", dest="debug", help=argparse.SUPPRESS, action="store_true", default=False
    )
    optional.add_argument(
        "--quiet", dest="quiet", help=argparse.SUPPRESS, action="store_true", default=False
    )
    parser._action_groups.append(optional)
    return parser


def sage_workflow(
    data,
    tes,
    out_dir=".",
    mask=None,
    prefix="",
    convention="bids",
    fittype="loglin",
    fitmode="all",
    combmode="t2s",
    debug=False,
    quiet=False,
):

    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)


    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    catd, ref_img = io.load_data(data, n_echos=n_echos)
    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        make_figures=False,
    )
    _, n_echos, _ = catd.shape

    if fitmode == "all":
        (t2star_map, s0_I_map, t2_map, s0_II_map) = decay.fit_decay_sage(catd, tes, fittype)
    else:
        (t2star_map, s0_I_map, t2_map, s0_II_map) = decay.fit_decay_ts_sage(catd, tes, fittype)

    # set a hard cap for the T2* map/timeseries
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2star_map.flatten(), 99.5, interpolation_method="lower")
    # cap_t2s_sec = utils.millisec2sec(cap_t2s * 10.0)
    # LGR.debug("Setting cap on T2* map at {:.5f}s".format(cap_t2s_sec))
    t2star_map[t2star_map > cap_t2s * 10] = cap_t2s

    # LGR.info("Computing optimal combination")
    # optimally combine data
    OCcatd = combine.make_optcom_sage(
        catd, tes, t2star_map, s0_I_map, t2_map, s0_II_map
    )  # REMOVED MASKSUM PARAMETER

    # clean up numerical errors
    for arr in (OCcatd, s0_I_map, t2star_map):
        np.nan_to_num(arr, copy=False)

    s0_I_map[s0_I_map < 0] = 0
    s0_II_map[s0_II_map < 0] = 0
    t2star_map[t2star_map < 0] = 0

    io_generator.save_file(
        utils.millisec2sec(t2star_map),
        "t2star img",
    )
    io_generator.save_file(s0_I_map, "s0_I img")
    io_generator.save_file(s0_II_map, "s0_II img")
    io_generator.save_file(
        utils.millisec2sec(t2star_map),
        "t2star img",
    )
    io_generator.save_file(
        utils.millisec2sec(t2_map),
        "t2 img",
    )
    io_generator.save_file(OCcatd, "combined img")

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "t2smap Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "tedana",
                "Version": __version__,
                "Description": (
                    "A pipeline estimating T2* from multi-echo fMRI data and "
                    "combining data across echoes."
                ),
                "CodeURL": "https://github.com/ME-ICA/tedana",
            }
        ],
    }
    io_generator.save_file(derivative_metadata, "data description json")


if __name__ == "__main__":
    options = _get_parser().parse_args()
    kwargs = vars(options)
    n_threads = kwargs.pop("n_threads")
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        sage_workflow(**kwargs)
