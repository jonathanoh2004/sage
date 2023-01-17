import os
import argparse
import shutil
import numpy as np
import nilearn.image


import tedana.io
import tedana.utils
from tedana.workflows import parser_utils
import config_sage
import parser_utils_sage


class Cmdline_Args:
    @staticmethod
    def parse_args():
        options = Cmdline_Args._get_parser().parse_args()
        kwargs = vars(options)
        return Cmdline_Args(**kwargs)

    def __init__(
        self,
        data_file_names,
        echo_times,
        out_dir="outputs",
        mask_file_name=None,
        convention="bids",
        prefix="",
        fittype="loglin",
        fitmode="all",
        combmode="t2s",
        gscontrol=None,
        tedpca="aic",
        tedort=False,
        fixed_seed=42,
        maxit=500,
        maxrestart=10,
        no_reports=False,
        png_cmap="coolwarm",
        low_mem=False,
        verbose=False,
        rerun_maps=None,
        debug=False,
        quiet=False,
        tslice=None,
        rerun_mixm=None,
        ctab=None,
        manacc=None,
        n_threads=-1,
    ):
        self.data_file_names = data_file_names
        self.echo_times = echo_times
        self.out_dir = out_dir
        self.mask_file_name = mask_file_name
        self.convention = convention
        self.prefix = prefix
        self.fittype = fittype
        self.fitmode = fitmode
        self.combmode = combmode
        self.gscontrol = gscontrol
        self.tedpca = tedpca
        self.tedort = tedort
        self.fixed_seed = fixed_seed
        self.maxit = maxit
        self.maxrestart = maxrestart
        self.no_reports = no_reports
        self.png_cmap = png_cmap
        self.low_mem = low_mem
        self.verbose = verbose
        self.rerun_maps = rerun_maps
        self.rerun_mixm = rerun_mixm
        self.debug = debug
        self.quiet = quiet
        self.tslice = tslice
        self.ctab = ctab
        self.manacc = manacc
        self.n_threads = n_threads

    @staticmethod
    def _get_parser():
        parser = argparse.ArgumentParser(prog="SAGEtedana")
        required = parser.add_argument_group("Required Arguments")
        required.add_argument(
            "-d",
            dest="data_file_names",
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
            dest="echo_times",
            nargs="+",
            metavar="TE",
            type=float,
            help="Echo times (in ms). E.g., 15.0 39.0 63.0",
            required=True,
        )
        parser.add_argument(
            "--mask",
            dest="mask_file_names",
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
            dest="fit_type",
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
            dest="fit_mode",
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
            dest="comb_mode",
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
            dest="rerun_maps_dir",
            metavar="PATH",
            type=lambda x: parser_utils_sage.is_valid_dir(parser, x),
            help=("Precalculated T2star, T2, S0I, and S0II maps and optcoms"),
            default=None,
        )
        parser.add_argument(
            "--tslice",
            dest="tslice",
            metavar="tstart:tend",
            type=lambda x: parser_utils_sage.is_valid_slice(parser, x),
            help=("Specify slice of the data in the fourth dimension"),
            default=None,
        )
        parser.add_argument(
            "--mix",
            dest="rerun_mixm",
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


def save_maps(img_maps, img_keys, io_generator):
    for img_map, img_key in zip(img_maps, img_keys, strict=True):
        output_key = config_sage.get_output_keys()[img_key]
        if output_key is None:
            raise ValueError("invalid output key")
        elif img_map is not None:
            io_generator.save_file(img_map, output_key)


def get_echo_times(tes):
    return np.array([float(te) for te in tes]) / 1000


def get_data(data, tes, tslice=None):
    if isinstance(data, str):
        data = [data]
    data, ref_img = tedana.io.load_data(data, n_echos=len(tes))

    if tslice is not None:
        data = _get_data_tslice(data, tslice)
    return data, ref_img


def get_mask(mask, data):
    if mask is not None:
        # load provided mask
        mask = nilearn.image.load_img(mask).get_fdata().reshape(-1).astype(bool)
    else:
        # include all voxels with at least one nonzero value
        mask = np.any(data != 0, axis=(1, 2)).reshape(config_sage.get_n_samps(data), 1)
    return mask


def get_gscontrol(gscontrol):
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]
    return gscontrol


def get_mixm(mixm, io_generator):
    if mixm is not None and os.path.isfile(mixm):
        mixm = os.path.abspath(mixm)
        # Allow users to re-run on same folder
        mixing_name = io_generator.get_name("ICA mixing tsv")
        if mixm != mixing_name:
            shutil.copyfile(mixm, mixing_name)
            shutil.copyfile(mixm, os.path.join(io_generator.out_dir, os.path.basename(mixm)))
        return mixm
    elif mixm is not None:
        raise IOError("Argument 'mixm' must be an existing file.")
    else:
        return None


def setup_loggers(repname, quiet, debug):
    tedana.utils.setup_loggers(logname=None, repname=repname, quiet=quiet, debug=debug)


def teardown_loggers():
    tedana.utils.teardown_loggers()


def gen_sub_dirs(sub_dirs):
    if not isinstance(sub_dirs, list):
        sub_dirs = [sub_dirs]
    nested_dir = None
    for sub_dir in sub_dirs:
        if sub_dir is not None:
            if nested_dir is None:
                nested_dir = os.path.abspath(sub_dir)
            else:
                nested_dir = os.path.join(nested_dir, sub_dir)
            if not os.path.isdir(nested_dir):
                os.mkdir(nested_dir)
    if nested_dir is None:
        raise ValueError("invalid subdirectories")
    return nested_dir


def get_io_generator(ref_img, convention, out_dir, prefix, verbose):
    io_generator = tedana.io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        verbose=verbose,
    )
    return io_generator


def get_rerun_maps(rerun, sub_dir, prefix, io_generator):
    rerun_keys = config_sage.get_rerun_keys()
    output_keys = config_sage.get_output_keys()
    rerun_imgs = {}
    if rerun is not None:
        if os.path.isdir(rerun):
            rerun_files = {
                k: os.path.join(rerun, sub_dir, prefix + io_generator.get_name(output_keys[k]))
                for k in rerun_keys
            }

            if not all([os.path.isfile(rerun_file) for rerun_file in rerun_files.values()]):
                raise ValueError(
                    "When rerunning, all rerun files must be present: " + str(rerun_keys)
                )

            for key, rerun_file in rerun_files.items():
                if os.path.isfile(rerun_file):
                    try:
                        rerun_imgs[key] = nilearn.image.load_img(rerun_file).get_fdata()
                    except Exception:
                        print("Error loading rerun imgs. Imgs will be recomputed.")
                        rerun_imgs.clear()
                        break
    return rerun_imgs


def _get_data_tslice(data, tslice):
    return data[:, :, tslice[0] : tslice[1]]


""" SCRAP
io_generator.save_file(s0_I_maps, get_output_key("s0I"))
    if s0_II_maps is not None:
        io_generator.save_file(s0_II_maps, get_output_key("s0II"))
    if delta_maps is not None:
        io_generator.save_file(delta_maps, get_output_key("delta"))
    io_generator.save_file(t2star_maps, get_output_key("t2star"))
    io_generator.save_file(t2_maps, get_output_key("t2"))
    if rmspe is not None:
        io_generator.save_file(rmspe, get_output_key("rmspe"))


"""
