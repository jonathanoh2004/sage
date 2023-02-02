# """
# gs_control_raw - called once with proper arguments if gsr in gscontrol, 0 otherwise
# minimum_image_regression - called once with proper arguments (if mir in gscontrol)
# tedpca - called once with proper arguments (if mixm is None)
# tedica - called once with proper arguments (if mixm is None)
# kundu_selection_v2 - called once with proper arguments
# compute_feats2 - called once with proper arguments
# get_metadata - called once with proper arguments
# writeresults - called once with proper arguments
# writeresults_echoes - called once with proper arguments (if verbose)
# denoise_ts - called once with proper arguments (if not no_report)
# carpet_plot - called once with proper arguments (if not no_report)
# comp_figures - called once with proper arguments (if not no_report)
# manual_selection - called once with proper arguments (if manacc is not None)
# generate_metrics - called once with proper arguments
# io_sage.save_maps - called with proper arguments: ([data_oc], [data_oc_label], io_generator)

# need to mock tedica to control the keep_restarting variable
# """
# import numpy as np
# from unittest.mock import MagicMock, patch

# from tedana.gscontrol import gscontrol_raw, minimum_image_regression
# from tedana.decomposition import tedpca, tedica
# from tedana.selection import kundu_selection_v2, manual_selection
# from tedana.stats import computefeats2
# from tedana.metrics.collect import generate_metrics, get_metadata
# from tedana.io import writeresults, writeresults_echoes, denoise_ts
# from tedana.reporting.static_figures import carpet_plot, comp_figures
# from tedana.workflows.sage.io_sage import save_maps

# from tedana.workflows.sage import denoise_sage, cmdline_sage, config_sage, io_sage


# @patch("tedana.gscontrol.gscontrol_raw")
# @patch("io_sage.save_maps")
# @patch("tedana.decomposition.tedpca")
# @patch("tedana.decomposition.tedica")
# @patch("tedana.metrics.collect.generate_metrics")  # same no times as tedica
# @patch("tedana.selection.kundu_selection_v2")  # same no times as tedica
# @patch("tedana.selection.manual_selection")
# @patch("tedana.stats.computefeats2")
# @patch("tedana.metrics.collect.get_metadata")
# @patch("tedana.io.writeresults")
# @patch("tedana.gscontrol.minimum_image_regression")
# @patch("tedana.io.writeresults_echoes")
# @patch("tedana.io.denoise_ts")
# @patch("tedana.reporting.static_figures.carpet_plot")
# @patch("tedana.reporting.static_figures.comp_figures")
# @patch("tedana.reporting.generate_report")
# def test_denoise(
#     mock_generatereport,
#     mock_compfigures,
#     mock_carpetplot,
#     mock_denoisets,
#     mock_writeresultsechoes,
#     mock_minimumimageregression,
#     mock_writeresults,
#     mock_getmetadata,
#     mock_computefeats2,
#     mock_manualselection,
#     mock_kunduselectionv2,
#     mock_generatemetrics,
#     mock_tedica,
#     mock_tedpca,
#     mock_savemaps,
#     mock_gscontrolraw,
# ):
#     repname = "repname"
#     bibtex_file = config_sage.get_bibtex_file()
#     repname = config_sage.get_repname()
#     data_oc = np.zeros((3, 3))
#     data_oc_label = "data_oc_label"
#     data = np.zeros((3, 3))
#     tes = np.array([1, 2, 3, 4, 5])
#     cmdline_args = cmdline_sage.Cmdline_Args(data_file_names=[], echo_times=[])
#     io_generator = io_sage.get_io_generator()
#     mask = io_sage.get_mask()
#     masksum = mask.astype(int)
#     gscontrol = ["gsr", "mir"]
#     mixm = np.zeros((3, 3))

#     denoise_sage.denoise(
#         data_oc,
#         data_oc_label,
#         io_generator,
#         data,
#         tes,
#         mask,
#         masksum,
#         gscontrol,
#         mixm,
#         repname,
#         bibtex_file,
#         cmdline_args,
#     )


# gsr in gscontrol, mixm - cmdline_args.ctab cmdline_args.manacc, cmdline_args.verbose, cmdline_args.tedort, mir in gscontrol, cmdline_args.no_reports
