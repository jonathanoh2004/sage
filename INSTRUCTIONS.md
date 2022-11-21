# README
git clone https://github.com/awstanton/tedana-forked.git
python -m venv env
env/Scripts/Activate.ps1
pip install -e .
python tedana\workflows\sage.py -d C:\Users\astanton005\Data\SAGE-testdata\Multigre_SAGE_e1_tshift_bet.nii.gz C:\Users\astanton005\Data\SAGE-testdata\Multigre_SAGE_e2_tshift_bet.nii.gz C:\Users\astanton005\Data\SAGE-testdata\Multigre_SAGE_e3_tshift_bet.nii.gz C:\Users\astanton005\Data\SAGE-testdata\Multigre_SAGE_e4_tshift_bet.nii.gz C:\Users\astanton005\Data\SAGE-testdata\Multigre_SAGE_e5_tshift_bet.nii.gz -e 7.9 27 58 77 96
deactivate