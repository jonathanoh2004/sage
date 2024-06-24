import nibabel as nib
import numpy as np

def check_niimg(file):
    img = nib.load(file)
    img.dataobj
    data = img.get_fdata()
    data.shape
    print(data)  # Print data to verify non-zero values
    return img

def reshape_niimg(data):
    if isinstance(data, (str, nib.spatialimages.SpatialImage)):
        img = check_niimg(data)
        data = img.get_fdata()
        data *= 100  # Apply scaling if necessary

    fdata = data.reshape((-1,) + data.shape[3:]).squeeze()
    print(f"Reshaped data:\n{fdata}")

    return fdata

# ===========================================================================================
file_path = '/Users/sudarshan/Documents/Jonathan/data/data2/sub-01_ses-1_task-rest_echo-2_bold.nii.gz'
img = reshape_niimg(file_path)

# Print the final data