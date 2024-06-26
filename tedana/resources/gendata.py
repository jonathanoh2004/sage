import os

import nibabel as nib
import numpy as np


data = np.arange(float(3 * 3 * 3 * 4)).reshape(3, 3, 3, 4) / 50

img1 = nib.Nifti1Image(data + 4, affine=np.eye(4))
img2 = nib.Nifti1Image(data + 3.7, affine=np.eye(4))
img3 = nib.Nifti1Image(data + 2.3, affine=np.eye(4))
img4 = nib.Nifti1Image(data + 1.2, affine=np.eye(4))
img5 = nib.Nifti1Image(data + 0.4, affine=np.eye(4))

cwd = os.path.dirname(__file__)

nib.save(img1, cwd + "/gendata/img1.nii.gz")
nib.save(img2, cwd + "/gendata/img2.nii.gz")
nib.save(img3, cwd + "/gendata/img3.nii.gz")
nib.save(img4, cwd + "/gendata/img4.nii.gz")
nib.save(img5, cwd + "/gendata/img5.nii.gz")
