# README
git clone https://github.com/awstanton/tedana-forked.git
python -m venv env
# for windows, the below line should be:
#   env/Scripts/Activate.ps1
source env/bin/activate
pip install -e .
python tedana\workflows\sage.py -d {RootPath}\SAGE-testdata\Multigre_SAGE_e1_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e2_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e3_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e4_tshift_bet.nii.gz {RootPath}\SAGE-testdata\Multigre_SAGE_e5_tshift_bet.nii.gz -e 7.9 27 58 77 96
deactivate