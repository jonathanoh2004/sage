import os.path
import tedana.bibtex
from tedana.workflows.sage import nonlinear_3param_sage, nonlinear_4param_sage, loglinear_sage
from tedana.workflows.sage.combine_sage import make_optcom_sage


def get_fittypes():
    return ["loglinear", "nonlinear3", "nonlinear4"]


def get_keys_maps():
    return ["t2star", "s0I", "t2", "s0II", "delta", "rmspe"]


def get_keys_maps_nonlin_3param():
    return ["t2star", "s0I", "t2", "delta", "rmspe"]


def get_keys_maps_nonlin_4param():
    return ["t2star", "s0I", "t2", "s0II", "rmspe"]


def get_keys_maps_results():
    return ["r2star_res", "s0I_res", "r2_res", "s0II_res", "delta_res", "rmspe_res"]


def get_keys_maps_results_nonlin_3param():
    return ["r2star_res", "s0I_res", "r2_res", "delta_res", "rmspe_res"]


def get_keys_maps_results_nonlin_3param_short():
    return ["r2star_res", "s0I_res", "r2_res", "rmspe_res"]


def get_keys_maps_results_nonlin_4param():
    return ["r2star_res", "s0I_res", "r2_res", "s0II_res", "rmspe_res"]


def get_keys_maps_guesses():
    return ["r2star_guess", "s0I_guess", "r2_guess", "s0II_guess", "delta_guess"]


def get_keys_maps_guesses_nonlin_3param():
    return ["r2star_guess", "s0I_guess", "r2_guess", "delta_guess"]


def get_keys_maps_guesses_nonlin_4param():
    return ["r2star_guess", "s0I_guess", "r2_guess", "s0II_guess"]


def get_keys_optcoms():
    return ["optcom t2star", "optcom t2"]


def get_keys_output():
    output_keys = {
        key: " ".join((key, "img"))
        for key in ["t2star", "t2", "optcom t2star", "optcom t2", "s0I", "s0II", "delta", "rmspe"]
    }
    return output_keys


def get_keys_rerun():
    return ["t2star", "t2", "optcom t2star", "optcom t2"]


def get_keys_shr_mem():
    """
    Shared memory keys
    Keys used by dicts to refer to shared memory names and arrays
    """
    return [
        "Y",
        "X",
        "r2star_guess",
        "s0I_guess",
        "r2_guess",
        "s0II_guess",
        "delta_res",
        "r2star_res",
        "s0I_res",
        "r2_res",
        "s0II_res",
        "rmspe_res",
    ]


def get_required_metrics():
    """
    Used to specify metrics computed by tedana denoising functions
    """
    return [
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


def get_func_maps(fittype):
    """
    Maps command line arguments to functions for
    computing T2* and T2 maps
    """
    if fittype == "loglin":
        return loglinear_sage.get_maps_loglinear
    elif fittype == "nonlin3":
        return nonlinear_3param_sage.get_maps_nonlinear_3param
    elif fittype == "nonlin4":
        return nonlinear_4param_sage.get_maps_nonlinear_4param
    else:
        raise ValueError("invalid fittype")


def get_optcom_func():
    return make_optcom_sage


def get_shape_maps(data):
    return (get_n_samps(data), get_n_vols(data))


def get_n_samps(data):
    return data.shape[0]


def get_n_echos(data):
    return data.shape[1]


def get_n_vols(data):
    return data.shape[2]


def get_dim_samps():
    return 0


def get_dim_echos():
    return 1


def get_dim_vols():
    return 2


def get_subdir(fittype):
    return fittype


def get_repname(sub_dir_tedana):
    return os.path.join(sub_dir_tedana, "report.txt")


def get_description_references(report):
    return tedana.bibtex.get_description_references(report)


def get_bibtex_file(sub_dir_tedana):
    return os.path.join(sub_dir_tedana, "references.bib")
