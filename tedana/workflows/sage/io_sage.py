import os
import argparse
import nilearn.image
from tedana.workflows import parser_utils
import numpy as np
from tedana import io


def get_echo_times(tes):
    return np.array([float(te) for te in tes]) / 1000

def get_data_tslice(data, tslice):
    return data[:, :, tslice[0] : tslice[1]]

def get_data(data, tes, tslice=None):
    if isinstance(data, str):
        data = [data]
    data, ref_img = io.load_data(data, n_echos=len(tes))

    if tslice is not None:
        data = get_data_tslice(data, tslice)
    return data, ref_img

def get_mask(mask, data):
    if mask is not None:
        # load provided mask
        mask = nilearn.image.load_img(mask).get_fdata().reshape(-1).astype(bool)
    else:
        # include all voxels with at least one nonzero value
        mask = np.any(data != 0, axis=(1, 2)).reshape(n_samps, 1)
    return mask

def get_gscontrol(gscontrol):
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]
    return gscontrol



def gen_subdir(sub_dirs):
    nested_dir = None
    for sub_dir in sub_dirs:
        if nested_dir is None:
            nested_dir = os.path.abspath(sub_dir)
        else:
            nested_dir = os.path.join(nested_dir, sub_dir)
        if not os.path.isdir(nested_dir):
            os.mkdir(nested_dir)
    return nested_dir

# used to get keys expected by io_generator.save_file() and outputs.json
output_keys = {
    key: " ".join((key, "img"))
    for key in ["t2star", "t2", "optcom_t2star", "optcom_t2", "s0I", "s0II", "delta", "rmspe"]
}

rerun_keys = ["t2star", "t2", "optcom_t2star", "optcom_t2"]

def get_output_key(key):
    return output_keys[key]


def get_rerun_maps(rerun=None, sub_dir, prefix, io_generator):
    if rerun is not None:
        if os.path.isdir(rerun):
            rerun_files = {
                k: os.path.join(rerun, sub_dir, prefix + io_generator.get_name(output_keys[k]))
                for k in rerun_keys
            }

            if not all([os.path.isfile(rerun_file) for rerun_file in rerun_files.values()]):
                raise ValueError(
                    "When rerunning, all relevant files must be present: T2star, T2, Optcom_T2star, and Optcom_T2"
                )

            try:
                rerun_imgs = {
                    k: nilearn.image.load_img(rerun_files[k]).get_fdata() for k in rerun_files
                }
            except Exception:
                print("Error loading rerun imgs. Imgs will be recomputed.")
                rerun_imgs = None
        else:
            rerun_imgs = None

        if rerun_imgs is not None:
            t2star_maps = rerun_imgs["t2star"].reshape(n_samps, n_vols)
            t2_maps = rerun_imgs["t2"].reshape(n_samps, n_vols)
            optcom_t2star = rerun_imgs["optcom_t2star"].reshape(n_samps, n_vols)
            optcom_t2 = rerun_imgs["optcom_t2"].reshape(n_samps, n_vols)
        else:
            t2star_maps, t2_maps, optcom_t2star, optcom_t2 = None, None, None, None

        return t2star_maps, t2_maps, optcom_t2star, optcom_t2


def get_parser():
    parser = argparse.ArgumentParser(prog="SAGEtedana")
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-d",
        dest="data",
        nargs="+",
        metavar="FILE",
        type=lambda x: parser_utils.is_valid_file(parser, x),
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
    parser.add_argument(
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: parser_utils.is_valid_file(parser, x),
        help=(
            "Binary mask of voxels to include in TE "
            "Dependent Analysis. Must be in the same "
            "space as `data`."
        ),
        default=None,
    )
    parser.add_argument(
        "--fittype",
        dest="fittype",
        action="store",
        choices=["loglin", "nonlin3", "nonlin4"],
        help="Desired Fitting Method"
        '"loglin" means that a linear model is fit'
        " to the log of the data (default)"
        '"nonlin" means that a more computationally'
        "demanding monoexponential model is fit"
        "to the raw data",
        default="loglin",
    )
    parser.add_argument(
        "--fitmode",
        dest="fitmode",
        action="store",
        choices=["all", "each"],
        help=(
            "Monoexponential model fitting scheme. "
            '"all" means that the model is fit, per voxel, '
            "across all timepoints. "
            '"each" means that the model is fit, per voxel '
            "and per timepoint."
        ),
        default="all",
    )
    parser.add_argument(
        "--combmode",
        dest="combmode",
        action="store",
        choices=["t2s", "paid"],
        help=("Combination scheme for TEs: t2s (Posse 1999, default), paid (Poser)"),
        default="t2s",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default="outputs",
    )
    parser.add_argument(
        "--prefix", dest="prefix", type=str, help="Prefix for filenames generated.", default=""
    )
    parser.add_argument(
        "--convention",
        dest="convention",
        action="store",
        choices=["orig", "bids"],
        help=("Filenaming convention. bids will use the latest BIDS derivatives version."),
        default="bids",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--debug", dest="debug", help=argparse.SUPPRESS, action="store_true", default=False
    )
    parser.add_argument(
        "--quiet", dest="quiet", help=argparse.SUPPRESS, action="store_true", default=False
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Generate intermediate and additional files.",
        default=False,
    )
    parser.add_argument(
        "--rerun-dir",
        dest="rerun",
        metavar="PATH",
        type=lambda x: parser_utils.is_valid_dir(parser, x),
        help=("Precalculated T2star, T2, S0I, and S0II maps and optcoms"),
        default=None,
    )
    parser.add_argument(
        "--tslice",
        dest="tslice",
        metavar="tstart:tend",
        type=lambda x: parser_utils.is_valid_slice(parser, x),
        help=("Specify slice of the data in the fourth dimension"),
        default=None,
    )
    parser.add_argument(
        "--mix",
        dest="mixm",
        metavar="FILE",
        type=lambda x: parser_utils.is_valid_file(parser, x),
        help=("File containing mixing matrix. If not provided, ME-PCA & ME-ICA is done."),
        default=None,
    )
    parser.add_argument(
        "--ctab",
        dest="ctab",
        metavar="FILE",
        type=lambda x: parser_utils.is_valid_file(parser, x),
        help=(
            "File containing a component table from which "
            "to extract pre-computed classifications. "
            "Requires --mix."
        ),
        default=None,
    )
    parser.add_argument(
        "--manacc",
        dest="manacc",
        metavar="INT",
        type=int,
        nargs="+",
        help=("List of manually accepted components. Requires --ctab and --mix."),
        default=None,
    )
    return parser
