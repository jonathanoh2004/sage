import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def check(file):
    
    # try and catch that looks at the slice at 1/4 2/4 3/4
    try:
        img = nib.load(file)                                        # load the file into variable
        data = img.get_fdata()                                      # get the fMRI data into our data variable
        print(f"Data shape: {data.shape}")                          # the 3D dimensions of our data will be printed here
        

        print("Sample data (1/4):")                                 # our 1/4 slice
        of_index = data.shape[-1] // 4
        print(data[..., of_index]) 
        
        print("Sample data (middle slice):")                        # our middle slice
        m_index = data.shape[-1] // 2                               
        print(data[..., m_index])
        
        print("Sample data (3/4):") 
        tf_index = data.shape[-1] // 4 * 3                          # our 3/4 slice
        print(data[..., tf_index])
        
        # used mpl to create some figures 
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(data[..., of_index], cmap='afmhot')          # change back to 'grey' for the standard look
        axes[0].set_title('1/4 slice')
        axes[1].imshow(data[..., m_index], cmap='afmhot')
        axes[1].set_title('Middle slice')
        axes[2].imshow(data[..., tf_index], cmap='afmhot')
        axes[2].set_title('3/4 slice')
        plt.show()

        return img                                                  # return the loaded data back to reshape
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def reshape(data):                                                  # loads the data from the path
    if isinstance(data, (str, nib.spatialimages.SpatialImage)):     # if there is no data then return none
        img = check(data)
        if img is None:
            return None
        data = img.get_fdata()                                      # else, get the data from the file indicated by the path

    try:
        fdata = data.reshape((-1,) + data.shape[3:]).squeeze()      # 
        print(f"Reshaped data shape: {fdata.shape}")
        print("Sample reshaped data:")
        print(fdata[0])                                             # Print a slice of the reshaped data for verification
        return fdata
    except Exception as e:
        print(f"Error reshaping data: {e}")
        return None

# ===========================================================================================
file_path = '/Users/sudarshan/Documents/Jonathan/data/data1/Multigre_SAGE_e1_tshift_tmean_bet.nii.gz'
fdata = reshape(file_path)
print("Final reshaped data:")
print(fdata)
