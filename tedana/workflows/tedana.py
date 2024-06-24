"""
Run the "canonical" TE-Dependent ANAlysis workflow.
"""


import argparse
import datetime
import json
import logging
import os
import os.path as op
import shutil
import sys
from glob import glob

import numpy as np
import pandas as pd
from nilearn.masking import compute_epi_mask
from scipy import stats
from threadpoolctl import threadpool_limits
from tedana import (
    __version__,
    combine,
    decay,
    decomposition,
    io,
    metrics,
    reporting,
    selection,
    utils,
    gscontrol as gsc,
)
from tedana.bibtex import get_description_references
from tedana.stats import computefeats2
from tedana.workflows.parser_utils import check_tedpca_value, is_valid_file

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

def tedana_workflow(
    data,
    tes,
    out_dir=".",
    mask=None,
    convention="bids",
    prefix="",
    fittype="loglin",
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
    t2smap=None,
    mixm=None,
    ctab=None,
    manacc=None,
):

    RepLGR.info(
        "TE-dependence analysis was performed on input data using the tedana workflow "
        "\\citep{dupre2021te}."
    )

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "tedana Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "tedana",
                "Version": __version__,
                "Description": (
                    "A denoising pipeline for the identification and removal "
                    "of non-BOLD noise from multi-echo fMRI data."
                ),
                "CodeURL": "https://github.com/ME-ICA/tedana",
            }
        ],
    }

    RepLGR.info(
        "This workflow used numpy \\citep{van2011numpy}, scipy \\citep{virtanen2020scipy}, "
        "pandas \\citep{mckinney2010data,reback2020pandas}, "
        "scikit-learn \\citep{pedregosa2011scikit}, "
        "nilearn, bokeh \\citep{bokehmanual}, matplotlib \\citep{Hunter:2007}, "
        "and nibabel \\citep{brett_matthew_2019_3233118}."
    )

    RepLGR.info(
        "This workflow also used the Dice similarity index "
        "\\citep{dice1945measures,sorensen1948method}."
    )
