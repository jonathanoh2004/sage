import argparse
import tedana.workflows.parser_utils
from tedana.workflows.sage import parser_utils_sage
from tedana import __version__


class Cmdline_Args:
    """
    Type used to parse arguments from command line
    and hold arguments for use by other modules.
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
        mask_file_name=None,
        mask_type="compute_epi_mask",
        clean_maps_tedana=False,
        fittype="loglin",
        out_dir="outputs",
        prefix="",
        rerun_maps_dir=None,
        rerun_mixm=None,
        ctab=None,
        manacc=None,
        gscontrol=None,
        tedort=False,
        tedpca="aic",
        maxit=500,
        maxrestart=10,
        fixed_seed=42,
        n_procs=-1,
        n_threads=1,
        tslice=None,
        low_mem=False,
        png_cmap="coolwarm",
        debug=None,
        quiet=None,
        verbose=False,
        no_reports=False,
        convention="bids",
    ):
        self.data_files_names = data_files_names
        self.echo_times = echo_times
        self.mask_file_name = mask_file_name
        self.mask_type = mask_type
        self.clean_maps_tedana = clean_maps_tedana
        self.fittype = fittype
        self.out_dir = out_dir
        self.prefix = prefix
        self.rerun_maps_dir = rerun_maps_dir
        self.rerun_mixm = rerun_mixm
        self.ctab = ctab
        self.manacc = manacc
        self.gscontrol = gscontrol
        self.tedort = tedort
        self.tedpca = tedpca
        self.maxit = maxit
        self.maxrestart = maxrestart
        self.fixed_seed = fixed_seed
        self.n_procs = n_procs
        self.n_threads = n_threads
        self.tslice = tslice
        self.low_mem = low_mem
        self.png_cmap = png_cmap
        self.debug = debug
        self.quiet = quiet
        self.verbose = verbose
        self.no_reports = no_reports
        self.convention = convention

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
                "order as the TEs are listed in the -e argument."
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
            "--mask-file-name",
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
            "--mask-type",
            dest="mask_type",
            action="store",
            choices=[
                "custom",
                "custom_restricted",
                "tedana",
                "tedana_adaptive",
                "compute_epi_mask",
                "none",
            ],
            help=(
                'The type of mask to use. A mask_type of either "custom" or "custom_restricted" '
                "means that the provided mask will be used without modification. A mask_type of "
                '"custom_restricted" causes the map to be restricted to the T2* and T2 maps after '
                'computing them. A mask_type of "tedana" or "tedana_adaptive" means that the mask '
                "will be computed the same way tedana computes it. If a mask is provided, the "
                'resulting mask will be more restrictive. A mask_type of "tedana_adaptive" means '
                "that the masksum used in PCA will be restricted to those echos that tedana "
                'deems as "good". A mask_type of "compute_epi_mask" computes the mask using a '
                'function from the nilearn library. A mask_type of "none" means that the mask '
                'will be all-inclusive. With a mask_type of "custom" or "custom_restricted", a '
                "mask must be provided."
            ),
            default="compute_epi_mask",
        )
        parser.add_argument(
            "--clean-maps-tedana",
            dest="clean_maps_tedana",
            action="store_true",
            help="Apply data cleaning procedure used by tedana to computed maps.",
            default=False,
        )
        parser.add_argument(
            "--fittype",
            dest="fittype",
            action="store",
            choices=["loglin", "nonlin3", "nonlin4"],
            help=(
                'Procedure used to compute maps. A fittype of "loglin" means that a '
                'linear model will be fit to the log of the data. A fittype of "nonlin3" '
                "means that a 3-parameter model (R2, R2*, and S0I) will be fit to the data by "
                'performing curve fitting. A fittype of "nonlin4" means that a 4-parameter '
                "(R2, R2*, S0I, and S0II) model will be fit to the data by performing curve "
                "fitting."
            ),
            default="loglin",
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
            "--rerun-dir",
            dest="rerun_maps_dir",
            metavar="PATH",
            type=lambda x: parser_utils_sage.is_valid_dir(parser, x),
            help=(
                "Use precalculated T2*, T2, S0I, and S0II maps and T2* and T2 optimal "
                "combinations instead of computing the maps and optimal combinations again."
            ),
            default=None,
        )
        parser.add_argument(
            "--mix",
            dest="rerun_mixm",
            metavar="FILE",
            type=lambda x: tedana.workflows.parser_utils.is_valid_file(parser, x),
            help=("Use precalculated mixing matrix instead of running ME-PCA and ME-ICA again."),
            default=None,
        )
        parser.add_argument(
            "--ctab",
            dest="ctab",
            metavar="FILE",
            type=lambda x: tedana.workflows.parser_utils.is_valid_file(parser, x),
            help=(
                "Use pre-generated component table with pre-computed classifications instead of "
                "generating the component table again. Requires --mix."
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
            "--gscontrol",
            dest="gscontrol",
            required=False,
            action="store",
            nargs="+",
            help=(
                "Perform additional denoising to remove spatially diffuse noise. This "
                "argument can be single value or a space-delimited list. Default is None."
            ),
            choices=["mir", "gsr"],
            default=None,
        )
        parser.add_argument(
            "--tedort",
            dest="tedort",
            action="store_true",
            help=(
                "Orthogonalize rejected components w.r.t. accepted components prior to denoising."
            ),
            default=False,
        )
        parser.add_argument(
            "--tedpca",
            dest="tedpca",
            type=tedana.workflows.parser_utils.check_tedpca_value,
            help=(
                "Method with which to select components in TEDPCA. "
                "PCA decomposition with the mdl, kic and aic options "
                "is based on a Moving Average (stationary Gaussian) "
                "process and are ordered from most to least aggressive. "
                "Users may also provide a float from 0 to 1, in which case"
                "components will be selected based on the cumulative variance "
                "explained or an integer greater than 1 in which case the "
                "specified number of components will be selected."
                'Default is "aic".'
            ),
            default="aic",
        )
        parser.add_argument(
            "--maxit",
            dest="maxit",
            metavar="INT",
            type=int,
            help=("Maximum number of iterations for ICA."),
            default=500,
        )
        parser.add_argument(
            "--maxrestart",
            dest="maxrestart",
            metavar="INT",
            type=int,
            help=(
                "Maximum number of attempts for ICA. If ICA "
                "fails to converge, the fixed seed will be "
                "updated and ICA will be run again. If "
                "convergence is achieved before maxrestart "
                "attempts, ICA will finish early."
            ),
            default=10,
        )
        parser.add_argument(
            "--seed",
            dest="fixed_seed",
            metavar="INT",
            type=int,
            help=(
                "Value used for random initialization of ICA "
                "algorithm. Set to an integer value for reproducible "
                "ICA results. Set to -1 for varying results across "
                "ICA calls. Default is 42."
            ),
            default=42,
        )
        parser.add_argument(
            "--n-procs",
            dest="n_procs",
            type=int,
            action="store",
            help=(
                "Number of cpu cores to use in computing maps using nonlinear model. "
                "If this value is -1 or 0 or greater than the number of CPU cores, "
                "then the number of cores present will be used."
            ),
            default=-1,
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
            "--tslice",
            dest="tslice",
            metavar="tstart:tend",
            type=lambda x: parser_utils_sage.is_valid_slice(parser, x),
            help=("Specify slice of the data in the fourth dimension"),
            default=None,
        )
        parser.add_argument(
            "--lowmem",
            dest="low_mem",
            action="store_true",
            help=(
                "Enables low-memory processing, including the use "
                "IncrementalPCA. May increase workflow duration."
            ),
            default=False,
        )
        parser.add_argument(
            "--png-cmap",
            dest="png_cmap",
            type=str,
            help="Colormap for figures",
            default="coolwarm",
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
            "--no-reports",
            dest="no_reports",
            action="store_true",
            help=(
                "Create a figures folder with static component maps, "
                "timecourse plots, and other diagnostic images, and "
                "display these in an interactive reporting framework."
            ),
            default=False,
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
            "-v", "--version", action="version", version="tedana v{}".format(__version__)
        )

        return parser
