"""
gs_control_raw - called once with proper arguments if gsr in gscontrol, 0 otherwise
minimum_image_regression - called once with proper arguments (if mir in gscontrol)
tedpca - called once with proper arguments (if mixm is None)
tedica - called once with proper arguments (if mixm is None)
kundu_selection_v2 - called once with proper arguments
compute_feats2 - called once with proper arguments
get_metadata - called once with proper arguments
writeresults - called once with proper arguments
writeresults_echoes - called once with proper arguments (if verbose)
denoise_ts - called once with proper arguments (if not no_report)
carpet_plot - called once with proper arguments (if not no_report)
comp_figures - called once with proper arguments (if not no_report)
manual_selection - called once with proper arguments (if manacc is not None)
generate_metrics - called once with proper arguments
io_sage.save_maps - called with proper arguments: ([data_oc], [data_oc_label], io_generator)

need to mock tedica to control the keep_restarting variable
"""

from unittest.mock import MagicMock

from tedana.gscontrol import gscontrol_raw, minimum_image_regression
from tedana.decomposition import tedpca, tedica
from tedana.selection import kundu_selection_v2, manual_selection
from tedana.stats import computefeats2
from tedana.metrics.collect import generate_metrics, get_metadata
from tedana.io import writeresults, writeresults_echoes, denoise_ts
from tedana.reporting.static_figures import carpet_plot, comp_figures
from tedana.workflows.sage.io_sage import save_maps


def test_denoise():
    pass
