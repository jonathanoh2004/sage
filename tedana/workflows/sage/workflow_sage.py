import logging
from threadpoolctl import threadpool_limits
import numpy as np
from tedana.workflows.sage import (
    combine_sage,
    io_sage,
    config_sage,
    denoise_sage,
    utils_sage,
    cmdline_sage,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def sage_workflow(cmdline_args):
    ########################################################################################
    ####################### RETRIEVE AND PREP DATA #########################################
    ########################################################################################

    tes = io_sage.get_echo_times(cmdline_args.echo_times)

    data, ref_img = io_sage.get_data(
        cmdline_args.data_files_names, tes, cmdline_args.tslice
    )

    gscontrol = io_sage.get_gscontrol(cmdline_args.gscontrol)

    n_samps, n_echos, n_vols = (
        config_sage.get_n_samps(data),
        config_sage.get_n_echos(data),
        config_sage.get_n_vols(data),
    )

    mask = io_sage.get_mask(cmdline_args.mask_file_name, data)

    sub_dir = io_sage.gen_sub_dirs(
        [cmdline_args.out_dir, config_sage.get_sub_dir(cmdline_args.fittype)]
    )

    io_generator = io_sage.get_io_generator(
        ref_img=ref_img,
        convention=cmdline_args.convention,
        out_dir=sub_dir,
        prefix=cmdline_args.prefix,
        verbose=cmdline_args.verbose,
    )

    ########################################################################################
    ##################################### MAPS #############################################
    ########################################################################################

    if cmdline_args.rerun_maps_dir is not None:
        rerun_imgs = io_sage.get_rerun_maps(
            cmdline_args.rerun_maps_dir, sub_dir, cmdline_args.prefix, io_generator
        )
        maps_t2star = rerun_imgs["t2star"].reshape(n_samps, n_vols)
        maps_t2 = rerun_imgs["t2"].reshape(n_samps, n_vols)
        optcom_t2star = rerun_imgs["optcom_t2star"].reshape(n_samps, n_vols)
        optcom_t2 = rerun_imgs["optcom_t2"].reshape(n_samps, n_vols)

    else:
        fit_func = config_sage.get_maps_func(cmdline_args.fittype)
        (
            maps_t2star,
            maps_s0_I,
            maps_t2,
            maps_s0_II,
            maps_delta,
            maps_rmspe,
        ) = fit_func(data, tes, mask.reshape(n_samps, 1), cmdline_args.n_procs)

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

        # optcom_t2star[~np.isfinite(optcom_t2star)] = 0
        # optcom_t2[~np.isfinite(optcom_t2star)] = 0

    ########################################################################################
    ####################### TEDANA DENOISING ###############################################
    ########################################################################################

    for data_oc, data_oc_label in zip([optcom_t2star, optcom_t2], config_sage.get_optcoms_keys()):

        sub_dir_tedana = io_sage.gen_sub_dirs([sub_dir, data_oc_label])

        repname = config_sage.get_repname(sub_dir_tedana)
        bibtex_file = config_sage.get_bibtex_file(sub_dir_tedana)

        utils_sage.setup_loggers(repname, cmdline_args.quiet, cmdline_args.debug)

        io_generator = io_sage.get_io_generator(
            ref_img,
            convention=cmdline_args.convention,
            out_dir=sub_dir_tedana,
            prefix=cmdline_args.prefix,
            verbose=cmdline_args.verbose,
        )

        masksum = mask * n_echos

        mixm = io_sage.get_mixm(cmdline_args.rerun_mixm, io_generator)

        denoise_sage.denoise(
            data_oc,
            data_oc_label,
            io_generator,
            data,
            tes,
            mask,
            masksum,
            gscontrol,
            mixm,
            repname,
            bibtex_file,
            cmdline_args,
        )

        utils_sage.teardown_loggers()


def _main(argv=None):
    cmdline_args = cmdline_sage.Cmdline_Args.parse_args()
    n_threads = None if cmdline_args.n_threads == -1 else cmdline_args.n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        sage_workflow(cmdline_args)


if __name__ == "__main__":
    _main()
