"""
Estimate T2, T2*, S0_I, and S0_II for SAGE ME-fMRI according to (cite paper)
and combine data across TEs according to (cite paper)
"""


import os
import sys
import logging
from threadpoolctl import threadpool_limits
import argparse
import json
import numpy as np
import pandas as pd
import nilearn.image
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

import tedana.workflows.parser_utils as parser_utils
from tedana.stats import computefeats2

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

def sage_workflow(
    data,
    tes,
    out_dir,
    mask=None,
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
    rerun=None,
    debug=False,
    quiet=False,
    mixm=None,
    ctab=None,
    manacc=None,
):
    ########################################################################################
    ####################### RETRIEVE AND PREP DATA #########################################
    ########################################################################################

    tes = np.array([float(te) for te in tes]) / 1000

    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]

    if isinstance(data, str):
        data = [data]

    catd, ref_img = io.load_data(data, n_echos=len(tes))

    n_samps, n_echos, n_vols = catd.shape

    # TODO: see how different masks work in the denoising section
    if mask is not None:
        # load provided mask
        mask = nilearn.image.load_img(mask).get_fdata().reshape(-1).astype(bool)
    else:
        # # include all voxels with at least one nonzero value
        mask = np.any(catd != 0, axis=(1, 2)).reshape(n_samps, 1)
        # first_echo_img = io.new_nii_like(io_generator.reference_img, catd[:, 0, :])
        # mask = compute_epi_mask(first_echo_img)

    # set up output directory structure
    out_dir = os.path.abspath("outputs")
    sub_dir = os.path.abspath(os.path.join("outputs", "_".join((fittype, fitmode))))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)

    # set up logging
    logname = None
    repname = os.path.join(out_dir, "report.txt")

    utils.setup_loggers(logname=logname, repname=repname, quiet=quiet, debug=debug)

    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=sub_dir,
        prefix=prefix,
        config="auto",
        verbose=verbose,
    )
    # used to get keys expected by io_generator.save_file and outputs.json
    output_keys = {
        "t2star": ' '.join(("t2star", "img")),
        "t2": ' '.join(("t2", "img")),
        "s0I": ' '.join(("s0I", "img")),
        "s0II": ' '.join(("s0II", "img")),
        "rmspe": ' '.join(("rmspe", "img")),
        "optcom_t2star": ' '.join(("optcom", "t2star", "img")),
        "optcom_t2": ' '.join(("optcom", "t2", "img")),
    }

    if rerun is not None:
        if os.path.isdir(rerun):
            rerun_files = {k: os.path.join(rerun, sub_dir, prefix + io_generator.get_name(output_keys[k])) for k in output_keys}
            if not all([os.path.isfile(rerun_file) for rerun_file in rerun_files.values()]):
                raise ValueError("When rerunning, all relevant files must be present: T2star, T2, S0I, S0II, optcom_T2star, and optcom_T2")
        try:
            rerun_imgs = {k: nilearn.image.load_img(rerun_files[k]).get_fdata() for k in rerun_files}
        except Exception:
            print("Error loading rerun imgs. Imgs will be recomputed.")
            rerun_imgs = None

        # TODO: change how the reshaping is done based on fitmode
        t2star_maps = rerun_imgs["t2star"].reshape(n_samps, n_vols)
        t2_maps = rerun_imgs["t2"].reshape(n_samps, n_vols)
        s0I_maps = rerun_imgs["s0I"].reshape(n_samps, n_vols)
        s0II_maps = rerun_imgs["s0II"].reshape(n_samps, n_vols)
        optcom_t2star = rerun_imgs["optcom_t2star"].reshape(n_samps, n_vols)
        optcom_t2 = rerun_imgs["optcom_t2"].reshape(n_samps, n_vols)
    else:
        ########################################################################################
        ####################### COMPUTE S0, T2*, T2 MAPS #######################################
        ########################################################################################

        # TODO: decide on which data cleaning procedures to use for computing maps
        # TODO: validate nonlinear decay fitting
        # TODO: make both loglinear and nonlinear work with both 1 or >1 time points
        # TODO: decide on how to do the bounds in nonlinear

        # If fitmode="all", each output map is over samples (S)
        # Else if fitmode="each", each output map is over samples and volumes (S x T)
        t2star_maps, s0_I_maps, t2_maps, delta_maps, rmspe = decay.fit_decay_sage(catd, tes, mask.reshape(n_samps, 1), fittype, fitmode)

        s0_II_maps = (1 / delta_maps) * s0_I_maps
        # s0_II_maps[~np.isfinite(s0_II_maps)] = 0

        ########################################################################################
        ####################### COMPUTE OPTIMAL COMBINATIONS ###################################
        ########################################################################################

        # TODO: decide whether to average over volumes here in the event of loglin fitting
        # TODO: check if changes are needed to optcom based on assumed model (i.e. 4 parameter fit with delta)
        # TODO: determine whether tese should be included in the weighting for T2-weighted
        # TODO: make work with single or varying numbers of time points
        # TODO: decide on masking

        optcom_t2star, optcom_t2 = combine.make_optcom_sage(
            catd, tes, t2star_maps, s0_I_maps, t2_maps, s0_II_maps, mask.reshape(n_samps, 1)
        )

        # TODO: decide on data cleaning steps to use for computing optimal combinations
        # np.nan_to_num(optcom_t2star, copy=False)
        # np.nan_to_num(optcom_t2, copy=False)

        # s0_I_map[s0_I_map < 0] = 0
        # s0_II_map[s0_II_map < 0] = 0
        # t2star_map[t2star_map < 0] = 0
        # t2_map[t2_map < 0] = 0
        
        ########################################################################################
        ####################### WRITE MAPS AND OPTCOMS OUTPUTS #################################
        ########################################################################################

        # t2star_outputs_key = ' '.join(("t2star", "img"))
        # t2_outputs_key = ' '.join(("t2", "img"))
        # s0I_outputs_key = ' '.join(("s0I", "img"))
        # s0II_outputs_key = ' '.join(("s0II", "img"))

        io_generator.save_file(rmspe, output_keys["rmspe"])
        io_generator.save_file(s0_I_maps, output_keys["s0I"])
        io_generator.save_file(s0_II_maps, output_keys["s0II"])
        io_generator.save_file(t2star_maps, output_keys["t2star"])
        io_generator.save_file(t2_maps, output_keys["t2"])

        ########################################################################################
        ####################### DENOISING ######################################################
        ########################################################################################

        # optcom_t2star_outputs_key = ' '.join(("optcom", "t2star", "img"))
        # optcom_t2_outputs_key = ' '.join(("optcom", "t2", "img"))

        io_generator.save_file(optcom_t2star, output_keys["optcom_t2star"])
        io_generator.save_file(optcom_t2, output_keys["optcom_t2"])

    for data_oc, iter_label in [(optcom_t2star, "optcom_t2star"), (optcom_t2, "optcom_t2")]:
        sub_dir_tedana = os.path.abspath(os.path.join(sub_dir, iter_label))
        if not os.path.isdir(sub_dir_tedana):
            os.mkdir(sub_dir_tedana)

        io_generator = io.OutputGenerator(
                ref_img,
                convention=convention,
                out_dir=sub_dir_tedana,
                prefix=prefix,
                config="auto",
                verbose=verbose,
            )

        # masksum = np.tile([n_echos], n_samps)
        masksum = mask * n_echos

        # regress out global signal unless explicitly not desired
        if "gsr" in gscontrol:
            catd, data_oc = gsc.gscontrol_raw(catd, data_oc, n_echos, io_generator)

        # Identify and remove thermal noise from data
        dd, n_components = decomposition.tedpca(
            catd,
            data_oc,
            combmode,
            mask,
            masksum,
            t2star_maps,
            io_generator,
            tes=tes,
            algorithm=tedpca,
            kdaw=10.0,
            rdaw=1.0,
            verbose=verbose,
            low_mem=low_mem,
        )
        if verbose:
            io_generator.save_file(utils.unmask(dd, mask), "whitened img")

        # Perform ICA, calculate metrics, and apply decision tree
        # Restart when ICA fails to converge or too few BOLD components found
        keep_restarting = True
        n_restarts = 0
        seed = fixed_seed

        while keep_restarting:
            mmix, seed = decomposition.tedica(
                dd, n_components, seed, maxit, maxrestart=(maxrestart - n_restarts)
            )
            seed += 1
            n_restarts = seed - fixed_seed

            # Estimate betas and compute selection metrics for mixing matrix
            # generated from dimensionally reduced data using full data (i.e., data
            # with thermal noise)
            print("Making second component selection guess from ICA results")
            required_metrics = [
                "kappa",
                "rho",
                "countnoise",
                "countsigFT2",
                "countsigFS0",
                "dice_FT2",
                "dice_FS0",
                "signal-noise_t",
                "variance explained",
                "normalized variance explained",
                "d_table_score",
            ]
            comptable = metrics.collect.generate_metrics(
                catd,
                data_oc,
                mmix,
                masksum,
                tes,
                io_generator,
                "ICA",
                metrics=required_metrics,
            )
            comptable, metric_metadata = selection.kundu_selection_v2(comptable, n_echos, n_vols)

            n_bold_comps = comptable[comptable.classification == "accepted"].shape[0]
            if (n_restarts < maxrestart) and (n_bold_comps == 0):
                print("No BOLD components found. Re-attempting ICA.")
            elif n_bold_comps == 0:
                print("No BOLD components found, but maximum number of restarts reached.")
                keep_restarting = False
            else:
                keep_restarting = False

        # Write out ICA files.
        comp_names = comptable["Component"].values
        mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
        io_generator.save_file(mixing_df, "ICA mixing tsv")
        betas_oc = utils.unmask(computefeats2(data_oc, mmix, mask), mask)
        io_generator.save_file(betas_oc, "z-scored ICA components img")

        # Save component table and associated json
        io_generator.save_file(comptable, "ICA metrics tsv")
        metric_metadata = metrics.collect.get_metadata(comptable)
        io_generator.save_file(metric_metadata, "ICA metrics json")

        decomp_metadata = {
            "Method": (
                "Independent components analysis with FastICA algorithm implemented by sklearn. "
            ),
        }
        for comp_name in comp_names:
            decomp_metadata[comp_name] = {
                "Description": "ICA fit to dimensionally-reduced optimally combined data.",
                "Method": "tedana",
            }
        with open(io_generator.get_name("ICA decomposition json"), "w") as fo:
            json.dump(decomp_metadata, fo, sort_keys=True, indent=4)

        if comptable[comptable.classification == "accepted"].shape[0] == 0:
            print("No BOLD components detected! Please check data and results!")
        
        mmix_orig = mmix.copy()
        if tedort:
            acc_idx = comptable.loc[
                ~comptable.classification.str.contains("rejected")
            ].index.values
            rej_idx = comptable.loc[comptable.classification.str.contains("rejected")
            ].index.values
            acc_ts = mmix[:, acc_idx]
            rej_ts = mmix[:, rej_idx]
            betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
            pred_rej_ts = np.dot(acc_ts, betas)
            resid = rej_ts - pred_rej_ts
            mmix[:, rej_idx] = resid
            comp_names = [
                io.add_decomp_prefix(comp, prefix="ica", max_value=comptable.index.max())
                for comp in comptable.index.values
            ]

            mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
            io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")

            print(
                "Rejected components' time series were then "
                "orthogonalized with respect to accepted components' time "
                "series."
            )

        io.writeresults(
            data_oc,
            mask=mask,
            comptable=comptable,
            mmix=mmix,
            n_vols=n_vols,
            io_generator=io_generator,
        )

        if "mir" in gscontrol:
            gsc.minimum_image_regression(
                data_oc, mmix, mask, comptable, io_generator
            )

        if verbose:
            io.writeresults_echoes(catd, mmix, mask, comptable, io_generator)

        # with open(repname, "r") as fo:
        #     report = [line.rstrip() for line in fo.readlines()]
        #     report = " ".join(report)

        # with open(repname, "w") as fo:
        #     fo.write(report)

        # # Collect BibTeX entries for cited papers
        # references = get_description_references(report)

        # with open(bibtex_file, "w") as fo:
        #     fo.write(references)

        if not no_reports:
            print("Making figures folder with static component maps and timecourse plots.")

            dn_ts, hikts, lowkts = io.denoise_ts(
                data_oc, mmix, mask, comptable
            )

            reporting.static_figures.carpet_plot(
                optcom_ts=data_oc,
                denoised_ts=dn_ts,
                hikts=hikts,
                lowkts=lowkts,
                mask=mask,
                io_generator=io_generator,
                gscontrol=gscontrol,
            )
            reporting.static_figures.comp_figures(
                data_oc,
                mask=mask,
                comptable=comptable,
                mmix=mmix_orig,
                io_generator=io_generator,
                png_cmap=png_cmap,
            )

            # TODO: add back in along with logging
            img_t_r = io_generator.reference_img.header.get_zooms()[-1]
            if img_t_r == 0:
                raise IOError(
                    "Dataset has a TR of 0. This indicates incorrect"
                    " header information. To correct this, we recommend"
                    " using this snippet:"
                    "\n"
                    "https://gist.github.com/jbteves/032c87aeb080dd8de8861cb151bff5d6"
                    "\n"
                    "to correct your TR to the value it should be."
                )

            if sys.version_info.major == 3 and sys.version_info.minor < 6:
                warn_msg = (
                    "Reports requested but Python version is less than "
                    "3.6.0. Dynamic reports will not be generated."
                )
                print(warn_msg)
            else:
                print("Generating dynamic report")
                reporting.generate_report(io_generator, tr=img_t_r)

    utils.teardown_loggers()
    print("Workflow completed")
    


def _get_parser():
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
        choices=["loglin", "nonlin"],
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
        "--rerun",
        dest="rerun",
        metavar="DIR",
        type=lambda x: parser_utils.is_valid_dir(parser, x),
        help=("Precalculated T2star, T2, S0I, and S0II maps and optcoms"),
        default=None,
    )
    return parser


if __name__ == "__main__":
    options = _get_parser().parse_args()
    kwargs = vars(options)
    n_threads = kwargs.pop("n_threads")
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        sage_workflow(**kwargs)
