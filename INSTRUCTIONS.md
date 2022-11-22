# README
# clone repository locally
git clone https://github.com/awstanton/tedana-forked.
# create virtual environment
python -m venv env
# activate virtual environment
#   for Windows Powershell, it should be:
#       env/Scripts/Activate.ps1
#   for Windows cmd, it should be:
#       env/Scripts/activate.bat
source env/bin/activate
# install dependencies from setup.py file
pip install -e .
# REPLACE {RootPath} WITH THE PATH TO YOUR DATA
python tedana\workflows\sage.py -d {RootPath}\SAGE-testdata\Multigre_SAGE_e1_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e2_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e3_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e4_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e5_tshift_bet.nii.gz -e 7.9 27 58 77 96
# deactivate virtual environment
deactivate