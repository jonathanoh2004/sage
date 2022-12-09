"""
Estimate T2, T2*, S0_I, and S0_II for SAGE ME-fMRI according to (cite paper)
and combine data across TEs according to (cite paper)
"""


import os
import sys
from threadpoolctl import threadpool_limits
import argparse
import json
import numpy as np
import pandas as pd
import nilearn.image
from scipy import stats
from tedana import (
    __version__,
    combine,
    decay,
    io,
    utils,
    gscontrol as gsc,
    decomposition,
    metrics,
    selection,
    reporting,
)
from tedana.workflows.parser_utils import is_valid_file
from tedana.stats import computefeats2


def _get_parser():
    """
    Parses command line inputs for sage tedana

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser()
    # Argument parser follow template provided by RalphyZ
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
        default=None,
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
        default="ts",
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
    out_dir="outputs",
    mask=None,
    convention="bids",
    prefix="",
    fittype="loglin",
    fitmode="ts",
    combmode="t2s",
    tedpca="aic",
    fixed_seed=42,
    maxit=500,
    maxrestart=10,
    tedort=False,
    gscontrol=None,
    no_reports=False,
    png_cmap="coolwarm",
    verbose=False,
    low_mem=False,
    debug=False,
    quiet=False,
    mixm=None,
    ctab=None,
    manacc=None,
):
    # ensure tes are in appropriate format
    tes = np.array([float(te) for te in tes]) / 1000

    # Coerce gscontrol to list
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]

    # TODO: change naming of data to datafiles or similar
    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    catd, ref_img = io.load_data(data, n_echos=len(tes))

    ########################################################################################
    ####################### MAPS AND COMBINATIONS ##########################################
    ########################################################################################

    n_samps, n_echos, n_vols = catd.shape

    if mask is None:
        # Note: this mask is used in this section as well as in the denoising section
        mask = np.any(catd != 0, axis=(1, 2))
    else:
        mask = nilearn.image.load_img(mask).get_fdata().reshape(catd.shape[0], 1)

    fitmode = "all"

    ### Fit Parameters ###
    # each result is in format (t2star_map, s0_I_map, t2_map, s0_II_map)
    # where there is one result if fitmode is "all" and otherwise
    # there is one result for each time point
    t2star_maps, s0_I_maps, t2_maps, delta_maps = decay.fit_decay_sage(
        catd, tes, mask, fittype, fitmode
    )

    # t2star_maps[np.logical_or(np.isnan(t2star_maps), np.isinf(t2star_maps))] = 0
    # s0_I_maps[np.logical_or(np.isnan(s0_I_maps), np.isinf(s0_I_maps))] = 0
    # t2_maps[np.logical_or(np.isnan(t2_maps), np.isinf(t2_maps))] = 0
    # delta_maps[np.logical_or(np.isnan(delta_maps), np.isinf(delta_maps))] = 0

    s0_II_maps = (1 / delta_maps) * s0_I_maps
    # s0_II_maps[~np.isfinite(s0_II_maps)] = 0

    # ### optimally combine data ###
    # optcom_t2star, optcom_t2 = combine.make_optcom_sage(
    #     catd, tes, t2star_map, s0_I_map, t2_map, s0_II_map, mask
    # )

    # # clean up numerical errors
    # for arr in (optcom_t2star, optcom_t2):
    #     np.nan_to_num(arr, copy=False)

    # s0_I_map[s0_I_map < 0] = 0
    # s0_II_map[s0_II_map < 0] = 0
    # t2star_map[t2star_map < 0] = 0
    # t2_map[t2_map < 0] = 0

    out_dir = os.path.abspath("outputs")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        verbose=verbose,
    )

    io_generator.save_file(s0_I_maps, "s0_I img")
    io_generator.save_file(s0_II_maps, "s0_II img")
    io_generator.save_file(t2star_maps, "t2star img")
    io_generator.save_file(t2_maps, "t2 img")
    # io_generator.save_file(utils.unmask(optcom_t2star, mask), "combined T2* img")
    # io_generator.save_file(utils.unmask(optcom_t2, mask), "combined T2 img")

    # ########################################################################################
    # ####################### DENOISING ######################################################
    # ########################################################################################

    # iter_data = [optcom_t2star, optcom_t2]
    # iter_labels = ["T2star", "T2"]

    # for data_oc, iter_label in zip(iter_data, iter_labels):
    #     out_dir = os.path.abspath(out_dir + "_" + iter_label)
    #     if not os.path.isdir(out_dir):
    #         os.mkdir(out_dir)

    #     io_generator = imageio.OutputGenerator(
    #             ref_img,
    #             convention=convention,
    #             out_dir=out_dir,
    #             prefix=prefix,
    #             config="auto",
    #             verbose=verbose,
    #         )

    #     # mask = np.tile([True], data_oc_t2star_I.shape[0])
    #     masksum = np.tile([n_echos], n_samps)

    #     # regress out global signal unless explicitly not desired
    #     if "gsr" in gscontrol:
    #         catd, data_oc = gsc.gscontrol_raw(catd, data_oc, n_echos, io_generator)

    #     # Identify and remove thermal noise from data
    #     dd, n_components = decomposition.tedpca(
    #         catd,
    #         utils.unmask(data_oc, mask),
    #         combmode,
    #         mask,
    #         masksum,
    #         t2star_map,
    #         io_generator,
    #         tes=tes,
    #         algorithm=tedpca,
    #         kdaw=10.0,
    #         rdaw=1.0,
    #         verbose=verbose,
    #         low_mem=low_mem,
    #     )
    #     if verbose:
    #         io_generator.save_file(utils.unmask(dd, mask), "whitened img")

    #     # Perform ICA, calculate metrics, and apply decision tree
    #     # Restart when ICA fails to converge or too few BOLD components found
    #     keep_restarting = True
    #     n_restarts = 0
    #     seed = fixed_seed

    #     while keep_restarting:
    #         mmix, seed = decomposition.tedica(
    #             dd, n_components, seed, maxit, maxrestart=(maxrestart - n_restarts)
    #         )
    #         seed += 1
    #         n_restarts = seed - fixed_seed

    #         # Estimate betas and compute selection metrics for mixing matrix
    #         # generated from dimensionally reduced data using full data (i.e., data
    #         # with thermal noise)
    #         required_metrics = [
    #             "kappa",
    #             "rho",
    #             "countnoise",
    #             "countsigFT2",
    #             "countsigFS0",
    #             "dice_FT2",
    #             "dice_FS0",
    #             "signal-noise_t",
    #             "variance explained",
    #             "normalized variance explained",
    #             "d_table_score",
    #         ]
    #         comptable = metrics.collect.generate_metrics(
    #             catd,
    #             utils.unmask(data_oc, mask),
    #             mmix,
    #             masksum,
    #             tes,
    #             io_generator,
    #             "ICA",
    #             metrics=required_metrics,
    #         )
    #         comptable, metric_metadata = selection.kundu_selection_v2(comptable, n_echos, n_vols)

    #         n_bold_comps = comptable[comptable.classification == "accepted"].shape[0]
    #         if (n_restarts < maxrestart) and (n_bold_comps == 0):
    #             print("No BOLD components found. Re-attempting ICA.")
    #         elif n_bold_comps == 0:
    #             print("No BOLD components found, but maximum number of restarts reached.")
    #             keep_restarting = False
    #         else:
    #             keep_restarting = False

    #     # Write out ICA files.
    #     comp_names = comptable["Component"].values
    #     mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
    #     io_generator.save_file(mixing_df, "ICA mixing tsv")
    #     betas_oc = utils.unmask(computefeats2(utils.unmask(data_oc, mask), mmix, mask), mask)
    #     io_generator.save_file(betas_oc, "z-scored ICA components img")

    #     # Save component table and associated json
    #     io_generator.save_file(comptable, "ICA metrics tsv")
    #     metric_metadata = metrics.collect.get_metadata(comptable)
    #     io_generator.save_file(metric_metadata, "ICA metrics json")

    #     decomp_metadata = {
    #         "Method": (
    #             "Independent components analysis with FastICA algorithm implemented by sklearn. "
    #         ),
    #     }
    #     for comp_name in comp_names:
    #         decomp_metadata[comp_name] = {
    #             "Description": "ICA fit to dimensionally-reduced optimally combined data.",
    #             "Method": "tedana",
    #         }
    #     with open(io_generator.get_name("ICA decomposition json"), "w") as fo:
    #         json.dump(decomp_metadata, fo, sort_keys=True, indent=4)

    #     if comptable[comptable.classification == "accepted"].shape[0] == 0:
    #         print("No BOLD components detected! Please check data and results!")
    #     mmix_orig = mmix.copy()
    #     if tedort:
    #         acc_idx = comptable.loc[
    #             ~comptable.classification.str.contains("rejected")
    #         ].index.values
    #         rej_idx = comptable.loc[comptable.classification.str.contains("rejected")].index.values
    #         acc_ts = mmix[:, acc_idx]
    #         rej_ts = mmix[:, rej_idx]
    #         betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
    #         pred_rej_ts = np.dot(acc_ts, betas)
    #         resid = rej_ts - pred_rej_ts
    #         mmix[:, rej_idx] = resid
    #         comp_names = [
    #             imageio.add_decomp_prefix(comp, prefix="ica", max_value=comptable.index.max())
    #             for comp in comptable.index.values
    #         ]
    #         mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
    #         io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")

    #     imageio.writeresults(
    #         utils.unmask(data_oc, mask),
    #         mask=mask,
    #         comptable=comptable,
    #         mmix=mmix,
    #         n_vols=n_vols,
    #         io_generator=io_generator,
    #     )

    #     if "mir" in gscontrol:
    #         gsc.minimum_image_regression(
    #             utils.unmask(data_oc, mask), mmix, mask, comptable, io_generator
    #         )

    #     if verbose:
    #         imageio.writeresults_echoes(catd, mmix, mask, comptable, io_generator)

    #     # with open(repname, "r") as fo:
    #     #     report = [line.rstrip() for line in fo.readlines()]
    #     #     report = " ".join(report)

    #     # with open(repname, "w") as fo:
    #     #     fo.write(report)

    #     # # Collect BibTeX entries for cited papers
    #     # references = get_description_references(report)

    #     # with open(bibtex_file, "w") as fo:
    #     #     fo.write(references)

    #     if not no_reports:
    #         dn_ts, hikts, lowkts = imageio.denoise_ts(
    #             utils.unmask(data_oc, mask), mmix, mask, comptable
    #         )

    #         reporting.static_figures.carpet_plot(
    #             optcom_ts=utils.unmask(data_oc, mask),
    #             denoised_ts=dn_ts,
    #             hikts=hikts,
    #             lowkts=lowkts,
    #             mask=mask,
    #             io_generator=io_generator,
    #             gscontrol=gscontrol,
    #         )
    #         reporting.static_figures.comp_figures(
    #             utils.unmask(data_oc, mask),
    #             mask=mask,
    #             comptable=comptable,
    #             mmix=mmix_orig,
    #             io_generator=io_generator,
    #             png_cmap=png_cmap,
    #         )

    # utils.teardown_loggers()


if __name__ == "__main__":
    options = _get_parser().parse_args()
    kwargs = vars(options)
    n_threads = kwargs.pop("n_threads")
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        sage_workflow(**kwargs)
