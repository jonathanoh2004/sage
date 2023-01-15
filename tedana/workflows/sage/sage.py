import os
import sys
import logging
import json
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
from tedana import (
    __version__,
    io,
    utils,
    gscontrol as gsc,
    decomposition,
    metrics,
    selection,
    reporting,
)
from tedana.stats import computefeats2
from tedana.bibtex import get_description_references
from tedana.workflows.sage import combine_sage, decay_sage, io_sage, config_sage

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def sage_workflow(
    data,
    tes,
    out_dir="outputs",
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
    tslice=None,
    mixm=None,
    ctab=None,
    manacc=None,
):
    ########################################################################################
    ####################### RETRIEVE AND PREP DATA #########################################
    ########################################################################################

    tes = io_sage.get_echo_times(tes)

    data, ref_img = io_sage.get_data(data, tes, tslice)

    gscontrol = io_sage.get_gscontrol(gscontrol)

    n_samps, n_echos, n_vols = (
        config_sage.get_n_samps(data),
        config_sage.get_n_echos(data),
        config_sage.get_n_vols(data),
    )

    mask = io_sage.get_mask(mask, data)

    sub_dir = io_sage.gen_sub_dirs([out_dir, config_sage.get_sub_dir(fittype)])

    io_generator = io_sage.get_io_generator(
        ref_img=ref_img,
        convention=convention,
        out_dir=sub_dir,
        prefix=prefix,
        verbose=verbose,
    )

    ########################################################################################
    ##################################### MAPS #############################################
    ########################################################################################

    if rerun is not None:
        rerun_imgs = io_sage.get_rerun_maps(rerun, sub_dir, prefix, io_generator)
        maps_t2star = rerun_imgs["t2star"].reshape(n_samps, n_vols)
        maps_t2 = rerun_imgs["t2"].reshape(n_samps, n_vols)
        optcom_t2star = rerun_imgs["optcom_t2star"].reshape(n_samps, n_vols)
        optcom_t2 = rerun_imgs["optcom_t2"].reshape(n_samps, n_vols)

    else:
        fit_func = config_sage.get_maps_func(fittype)
        (
            maps_t2star,
            maps_s0_I,
            maps_t2,
            maps_s0_II,
            maps_delta,
            maps_rmspe,
        ) = fit_func(data, tes, mask.reshape(n_samps, 1))

        io_sage.save_maps(
            [maps_t2star, maps_s0_I, maps_t2, maps_s0_II, maps_delta, maps_rmspe],
            config_sage.get_maps_keys(),
            io_generator,
        )

        ########################################################################################
        ############################ OPTIMAL COMBINATIONS ######################################
        ########################################################################################

        optcom_t2star, optcom_t2 = combine_sage.make_optcom_sage(
            data, tes, maps_t2star, maps_s0_I, maps_t2, maps_s0_II, mask.reshape(n_samps, 1)
        )

        io_sage.save_maps([optcom_t2star, optcom_t2], config_sage.get_optcoms_keys(), io_generator)

    ########################################################################################
    ####################### TEDANA DENOISING ###############################################
    ########################################################################################
    required_metrics = config_sage.get_required_metrics()
    output_keys = config_sage.get_output_keys()

    data_orig = data.copy()

    for data_oc, data_oc_label in zip([optcom_t2star, optcom_t2], config_sage.get_optcoms_keys()):
        sub_dir_tedana = os.path.abspath(os.path.join(sub_dir, data_oc_label))

        if not os.path.isdir(sub_dir_tedana):
            os.mkdir(sub_dir_tedana)

        # set up logging
        repname = os.path.join(sub_dir_tedana, "report.txt")

        utils.setup_loggers(logname=None, repname=repname, quiet=quiet, debug=debug)

        io_generator = io.OutputGenerator(
            ref_img,
            convention=convention,
            out_dir=sub_dir_tedana,
            prefix=prefix,
            config="auto",
            verbose=verbose,
        )

        # regress out global signal unless explicitly not desired
        if "gsr" in gscontrol:
            data, data_oc = gsc.gscontrol_raw(data_orig, data_oc, n_echos, io_generator)

        io_generator.save_file(data_oc, output_keys[data_oc_label])

        # masksum = np.tile([n_echos], n_samps)
        masksum = mask * n_echos

        # Identify and remove thermal noise from data
        dd, n_components = decomposition.tedpca(
            data,
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
            comptable = metrics.collect.generate_metrics(
                data,
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
            rej_idx = comptable.loc[comptable.classification.str.contains("rejected")].index.values
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
            gsc.minimum_image_regression(data_oc, mmix, mask, comptable, io_generator)

        if verbose:
            io.writeresults_echoes(data, mmix, mask, comptable, io_generator)

        with open(repname, "r") as fo:
            report = [line.rstrip() for line in fo.readlines()]
            report = " ".join(report)

        with open(repname, "w") as fo:
            fo.write(report)

        bibtex_file = os.path.join(sub_dir_tedana, "references.bib")

        # Collect BibTeX entries for cited papers
        references = get_description_references(report)

        with open(bibtex_file, "w") as fo:
            fo.write(references)

        if not no_reports:
            print("Making figures folder with static component maps and timecourse plots.")

            dn_ts, hikts, lowkts = io.denoise_ts(data_oc, mmix, mask, comptable)

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


def _main(argv=None):
    options = io_sage.get_parser().parse_args()
    kwargs = vars(options)
    n_threads = kwargs.pop("n_threads")
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        sage_workflow(**kwargs)


if __name__ == "__main__":
    _main()
