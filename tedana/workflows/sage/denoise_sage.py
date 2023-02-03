import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import tedana.gscontrol
import tedana.decomposition
import tedana.metrics
import tedana.selection
import tedana.reporting
import tedana.io
import tedana.stats
import tedana.utils
from tedana.workflows.sage import config_sage

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def denoise(
    data_oc,
    data_oc_label,
    io_generator,
    data,
    tes,
    mask,
    masksum,
    mask_clf,
    masksum_clf,
    gscontrol,
    mixm,
    repname,
    bibtex_file,
    cmdline_args,
):
    LGR.info("workflow begun for " + data_oc_label)
    required_metrics = config_sage.get_required_metrics()

    data_orig = data.copy()
    n_echos = config_sage.get_n_echos(data)
    n_vols = config_sage.get_n_vols(data)

    io_generator.save_file(masksum, "adaptive mask img")
    io_generator.save_file(masksum_clf, "adaptive mask clf img")

    # regress out global signal unless explicitly not desired
    if "gsr" in gscontrol:
        data, data_oc = tedana.gscontrol.gscontrol_raw(data_orig, data_oc, n_echos, io_generator)

    if mixm is None:

        # Identify and remove thermal noise from data
        dd, n_components = tedana.decomposition.tedpca(
            data,
            data_oc,
            None,
            mask_clf,
            masksum_clf,
            None,
            io_generator,
            tes=tes,
            algorithm=cmdline_args.tedpca,
            kdaw=10.0,
            rdaw=1.0,
            verbose=cmdline_args.verbose,
            low_mem=cmdline_args.low_mem,
        )
        if cmdline_args.verbose:
            io_generator.save_file(tedana.utils.unmask(dd, mask_clf), "whitened img")

        # Perform ICA, calculate metrics, and apply decision tree
        # Restart when ICA fails to converge or too few BOLD components found
        keep_restarting = True
        n_restarts = 0
        seed = cmdline_args.fixed_seed

        while keep_restarting:
            mmix, seed = tedana.decomposition.tedica(
                dd,
                n_components,
                seed,
                cmdline_args.maxit,
                maxrestart=(cmdline_args.maxrestart - n_restarts),
            )
            seed += 1
            n_restarts = seed - cmdline_args.fixed_seed

            # Estimate betas and compute selection metrics for mixing matrix
            # generated from dimensionally reduced data using full data (i.e., data
            # with thermal noise)
            LGR.info("Making second component selection guess from ICA results")
            comptable = tedana.metrics.collect.generate_metrics(
                data,
                data_oc,
                mmix,
                masksum_clf,
                tes,
                io_generator,
                "ICA",
                metrics=required_metrics,
            )
            comptable, metric_metadata = tedana.selection.kundu_selection_v2(
                comptable, n_echos, n_vols
            )

            n_bold_comps = comptable[comptable.classification == "accepted"].shape[0]
            if (n_restarts < cmdline_args.maxrestart) and (n_bold_comps == 0):
                LGR.warning("No BOLD components found. Re-attempting ICA.")
            elif n_bold_comps == 0:
                LGR.warning("No BOLD components found, but maximum number of restarts reached.")
                keep_restarting = False
            else:
                keep_restarting = False
    else:
        LGR.info("Using supplied mixing matrix from ICA")
        mixing_file = io_generator.get_name("ICA mixing tsv")
        mmix = pd.read_table(mixing_file).values

        if cmdline_args.ctab is None:
            comptable = tedana.metrics.collect.generate_metrics(
                data,
                data_oc,
                mmix,
                masksum_clf,
                tes,
                io_generator,
                "ICA",
                metrics=required_metrics,
            )
            comptable, metric_metadata = tedana.selection.kundu_selection_v2(
                comptable, n_echos, n_vols
            )
        else:
            LGR.info("Using supplied component table for classification")
            cmdline_args.ctab = os.path.abspath(cmdline_args.ctab)
            comptable = pd.read_table(cmdline_args.ctab)
            # Change rationale value of rows with NaN to empty strings
            comptable.loc[comptable.rationale.isna(), "rationale"] = ""

            if cmdline_args.manacc is not None:
                cmdline_args.manacc = [int(m) for m in cmdline_args.manacc]
                comptable, metric_metadata = tedana.selection.manual_selection(
                    comptable, acc=cmdline_args.manacc
                )

    # Write out ICA files.
    comp_names = comptable["Component"].values
    mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
    betas_oc = tedana.utils.unmask(tedana.stats.computefeats2(data_oc, mmix, mask), mask)
    metric_metadata = tedana.metrics.collect.get_metadata(comptable)

    io_generator.save_file(mixing_df, "ICA mixing tsv")
    io_generator.save_file(betas_oc, "z-scored ICA components img")
    io_generator.save_file(comptable, "ICA metrics tsv")
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
        LGR.warning("No BOLD components detected! Please check data and results!")

    mmix_orig = mmix.copy()

    if cmdline_args.tedort:
        tedort(comptable, mmix, io_generator)

    tedana.io.writeresults(
        data_oc,
        mask=mask,
        comptable=comptable,
        mmix=mmix,
        n_vols=n_vols,
        io_generator=io_generator,
    )

    if "mir" in gscontrol:
        tedana.gscontrol.minimum_image_regression(data_oc, mmix, mask, comptable, io_generator)

    if cmdline_args.verbose:
        tedana.io.writeresults_echoes(data, mmix, mask, comptable, io_generator)

    with open(repname, "r") as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = " ".join(report)

    with open(repname, "w") as fo:
        fo.write(report)

    try:
        description_references = config_sage.get_description_references(report)
        # Collect BibTeX entries for cited papers
        with open(bibtex_file, "w") as fo:
            fo.write(description_references)
    except FileNotFoundError:
        LGR.warning("Unable to find BibTeX references for citations.")

    if not cmdline_args.no_reports:
        LGR.info("Making figures folder with static component maps and timecourse plots.")

        dn_ts, hikts, lowkts = tedana.io.denoise_ts(data_oc, mmix, mask, comptable)

        tedana.reporting.static_figures.carpet_plot(
            optcom_ts=data_oc,
            denoised_ts=dn_ts,
            hikts=hikts,
            lowkts=lowkts,
            mask=mask,
            io_generator=io_generator,
            gscontrol=gscontrol,
        )
        tedana.reporting.static_figures.comp_figures(
            data_oc,
            mask=mask,
            comptable=comptable,
            mmix=mmix_orig,
            io_generator=io_generator,
            png_cmap=cmdline_args.png_cmap,
        )

        if sys.version_info.major == 3 and sys.version_info.minor < 6:
            LGR.warn(
                "Reports requested but Python version is less than "
                "3.6.0. Dynamic reports will not be generated."
            )
        else:
            LGR.info("Generating dynamic report")
            img_t_r = io_generator.reference_img.header.get_zooms()[-1]
            tedana.reporting.generate_report(io_generator, tr=img_t_r)
    LGR.info("workflow completed for " + data_oc_label)


def tedort(comptable, mmix, io_generator):
    acc_idx = comptable.loc[~comptable.classification.str.contains("rejected")].index.values
    rej_idx = comptable.loc[comptable.classification.str.contains("rejected")].index.values
    acc_ts = mmix[:, acc_idx]
    rej_ts = mmix[:, rej_idx]
    betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
    pred_rej_ts = np.dot(acc_ts, betas)
    resid = rej_ts - pred_rej_ts
    mmix[:, rej_idx] = resid
    comp_names = [
        tedana.io.add_decomp_prefix(comp, prefix="ica", max_value=comptable.index.max())
        for comp in comptable.index.values
    ]

    mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
    io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")

    RepLGR.info(
        "Rejected components' time series were then "
        "orthogonalized with respect to accepted components' time "
        "series."
    )
