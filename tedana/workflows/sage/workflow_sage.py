import logging
from threadpoolctl import threadpool_limits
from tedana.workflows.sage import (
    combine_sage,
    io_sage,
    config_sage,
    denoise_sage,
    utils_sage,
    cmdline_sage,
    clean_sage,
    masking_sage,
)

LGR = logging.getLogger("GENERAL")


def workflow_sage(cmdline_args):
    """
    The workflow function for SAGE (spin and gradient echo) fMRI sequences.
    1) Computes and outputs T2* and T2 maps.
    2) Computes and outputs T2* and T2 optimal combinations.
    3) Denoises T2* and T2 optimal combinations using tedana.
    """
    ########################################################################################
    ####################### RETRIEVE AND PREP DATA #########################################
    ########################################################################################

    tes = io_sage.get_echo_times(cmdline_args.echo_times)

    data, ref_img = io_sage.get_data(cmdline_args.data_files_names, tes, cmdline_args.tslice)

    gscontrol = io_sage.get_gscontrol(cmdline_args.gscontrol)

    n_samps, n_echos, n_vols = (
        config_sage.get_n_samps(data),
        config_sage.get_n_echos(data),
        config_sage.get_n_vols(data),
    )

    mask = io_sage.get_mask(data, cmdline_args.mask_type, cmdline_args.mask_file_name, ref_img)

    sub_dir = io_sage.get_sub_dir(cmdline_args)

    io_generator = io_sage.get_io_generator(
        ref_img=ref_img,
        convention=cmdline_args.convention,
        out_dir=sub_dir,
        prefix=cmdline_args.prefix,
        verbose=cmdline_args.verbose,
    )

    io_sage.check_header(io_generator)

    ########################################################################################
    ##################################### MAPS #############################################
    ########################################################################################

    if cmdline_args.rerun_maps_dir is not None:
        rerun_imgs = io_sage.get_rerun_maps(cmdline_args, ref_img)
        maps_t2star = rerun_imgs["t2star"].reshape(n_samps, n_vols)
        maps_t2 = rerun_imgs["t2"].reshape(n_samps, n_vols)
        optcom_t2star = rerun_imgs["optcom t2star"].reshape(n_samps, n_vols)
        optcom_t2 = rerun_imgs["optcom t2"].reshape(n_samps, n_vols)

    else:
        fit_func = config_sage.get_func_maps(cmdline_args.fittype)
        (
            maps_t2star,
            maps_s0I,
            maps_t2,
            maps_s0II,
            maps_delta,
            maps_rmspe,
        ) = fit_func(data, tes, mask.reshape(n_samps, 1), cmdline_args.n_procs)

        if cmdline_args.clean_maps_tedana:
            # maps should be modified without returning new arrays
            clean_sage.clean_maps_tedana(tes, maps_t2star, maps_t2, maps_s0I, maps_s0II)

        io_sage.save_maps(
            [maps_t2star, maps_s0I, maps_t2, maps_s0II, maps_delta, maps_rmspe],
            config_sage.get_keys_maps(),
            io_generator,
        )

        if cmdline_args.mask_type == "custom_restricted":
            masking_sage.restrict_mask(mask, maps_t2star, maps_t2)

        ########################################################################################
        ############################ OPTIMAL COMBINATIONS ######################################
        ########################################################################################

        optcom_t2star, optcom_t2 = combine_sage.make_optcom_sage(
            data, tes, maps_t2star, maps_s0I, maps_t2, maps_s0II, mask.reshape(n_samps, 1, 1)
        )

        clean_sage.clean_optcoms(optcom_t2star, optcom_t2)

        io_sage.save_maps([optcom_t2star, optcom_t2], config_sage.get_keys_optcoms(), io_generator)

    ########################################################################################
    ####################### TEDANA DENOISING ###############################################
    ########################################################################################

    masksum = mask * n_echos
    mask_clf, masksum_clf = masking_sage.get_adaptive_mask_clf(mask, masksum, data, cmdline_args)

    for data_oc, data_oc_label in zip([optcom_t2star, optcom_t2], config_sage.get_keys_optcoms()):

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

        mixm = io_sage.get_mixm(cmdline_args.rerun_mixm, io_generator)

        denoise_sage.denoise(
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
        )

        utils_sage.teardown_loggers()


def _main():
    cmdline_args = cmdline_sage.Cmdline_Args.parse_args()
    n_threads = None if cmdline_args.n_threads == -1 else cmdline_args.n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        workflow_sage(cmdline_args)


if __name__ == "__main__":
    _main()
