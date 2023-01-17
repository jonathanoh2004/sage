import nonlinear_3param_sage, nonlinear_4param_sage, nonlinear_sage, loglinear_sage
import os.path
from combine_sage import make_optcom_sage
import tedana.bibtex


def get_repname(sub_dir_tedana):
    return os.path.join(sub_dir_tedana, "report.txt")


def get_description_references(report):
    return tedana.bibtex.get_description_references(report)


def get_bibtex_file(sub_dir_tedana):
    return os.path.join(sub_dir_tedana, "references.bib")


def get_n_samps(data):
    return data.shape[0]


def get_n_echos(data):
    return data.shape[1]


def get_n_vols(data):
    return data.shape[2]


def get_fittypes():
    return ["loglinear", "nonlinear3", "nonlinear4"]


def get_maps_keys():
    return ["t2star", "s0I", "t2", "s0II", "delta", "rmspe"]


def get_optcoms_keys():
    return ["optcom_t2star", "optcom_t2"]


def get_required_metrics():
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


def get_output_keys():
    output_keys = {
        key: " ".join((key, "img"))
        for key in ["t2star", "t2", "optcom_t2star", "optcom_t2", "s0I", "s0II", "delta", "rmspe"]
    }
    return output_keys


def get_dim_samps():
    return 0


def get_dim_echos():
    return 1


def get_dim_vols():
    return 2


def get_rerun_keys():
    return ["t2star", "t2", "optcom_t2star", "optcom_t2"]


def get_optcom_func():
    return make_optcom_sage


def get_nonlinear_keys():
    return [
        "Y",
        "X",
        "r2star_guess",
        "s0_I_guess",
        "r2_guess",
        "s0_II_guess",
        "delta",
        "r2star_res",
        "s0_I_res",
        "r2_res",
        "s0_II_res",
        "rmspe_res",
    ]


def get_maps_func(fittype):
    if fittype == "loglin":
        return loglinear_sage.get_maps_loglinear
    elif fittype == "nonlin3":
        return nonlinear_3param_sage.get_maps_nonlinear_3param
    elif fittype == "nonlin4":
        return nonlinear_4param_sage.get_maps_nonlinear_4param
    else:
        raise ValueError("invalid fittype")


def get_sub_dir(fittype):
    return fittype
