import argparse
import tedana.workflows.parser_utils
from tedana.workflows.sage import parser_utils_sage


class Cmdline_Args:
    """
    Type used to parse arguments from command line
    and to hold arguments for retrieval.
    """

    @staticmethod
    def parse_args():
        options = Cmdline_Args._get_parser().parse_args()
        kwargs = vars(options)
        return Cmdline_Args(**kwargs)

    def __init__(
        self,
        data_files_names,
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
        rerun_maps_dir=None,
        rerun_mixm=None,
        debug=None,
        quiet=None,
        tslice=None,
        ctab=None,
        manacc=None,
        n_threads=1,
        n_procs=-1,
        clean_maps_tedana=False,
    ):
        self.data_files_names = data_files_names
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
        self.rerun_maps_dir = rerun_maps_dir
        self.rerun_mixm = rerun_mixm
        self.debug = debug
        self.quiet = quiet
        self.tslice = tslice
        self.ctab = ctab
        self.manacc = manacc
        self.n_threads = n_threads
        self.n_procs = n_procs
        self.clean_maps_tedana = clean_maps_tedana

    @staticmethod
    def _get_parser():
        parser = argparse.ArgumentParser(prog="SAGEtedana")
        required = parser.add_argument_group("Required Arguments")
        required.add_argument(
            "-d",
            dest="data_files_names",
            nargs="+",
            metavar="FILE",
            type=lambda x: tedana.workflows.parser_utils.is_valid_file(parser, x),
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
            dest="mask_file_name",
            metavar="FILE",
            type=lambda x: tedana.workflows.parser_utils.is_valid_file(parser, x),
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
            "--n-procs",
            dest="n_procs",
            type=int,
            action="store",
            help=(
                "Number of cpu cores to use in nonlinear fitting."
                "If this value is -1 or 0 or greater than the "
                "number of cores, then the number of cores found "
                "will be used."
            ),
            default=-1,
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
            type=lambda x: tedana.workflows.parser_utils.is_valid_file(parser, x),
            help=("File containing mixing matrix. If not provided, ME-PCA & ME-ICA is done."),
            default=None,
        )
        parser.add_argument(
            "--ctab",
            dest="ctab",
            metavar="FILE",
            type=lambda x: tedana.workflows.parser_utils.is_valid_file(parser, x),
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
        parser.add_argument(
            "--clean-maps-tedana",
            dest="clean_maps_tedana",
            action="store_true",
            help="Apply tedana cleaning to maps after computing them.",
            default=False,
        )
        return parser
