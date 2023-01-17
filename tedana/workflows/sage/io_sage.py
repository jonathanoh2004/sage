import os
import shutil
import numpy as np
import nilearn.image
import tedana.io
import tedana.utils
import config_sage


def get_echo_times(tes):
    return np.array([float(te) for te in tes]) / 1000


def get_data(data, tes, tslice=None):
    if isinstance(data, str):
        data = [data]
    data, ref_img = tedana.io.load_data(data, n_echos=len(tes))

    if tslice is not None:
        data = data[:, :, tslice[0] : tslice[1]]
    return data, ref_img


def get_mask(mask, data):
    if mask is not None:
        # load provided mask
        mask = nilearn.image.load_img(mask).get_fdata().reshape(-1).astype(bool)
    else:
        # include all voxels with at least one nonzero value
        mask = np.any(data != 0, axis=(1, 2)).reshape(config_sage.get_n_samps(data), 1)
    return mask


def get_gscontrol(gscontrol):
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]
    return gscontrol


def get_mixm(mixm, io_generator):
    if mixm is not None and os.path.isfile(mixm):
        mixm = os.path.abspath(mixm)
        # Allow users to re-run on same folder
        mixing_name = io_generator.get_name("ICA mixing tsv")
        if mixm != mixing_name:
            shutil.copyfile(mixm, mixing_name)
            shutil.copyfile(mixm, os.path.join(io_generator.out_dir, os.path.basename(mixm)))
        return mixm
    elif mixm is not None:
        raise IOError("Argument 'mixm' must be an existing file.")
    else:
        return None


def get_rerun_maps(rerun, sub_dir, prefix, io_generator):
    rerun_keys = config_sage.get_rerun_keys()
    output_keys = config_sage.get_output_keys()
    rerun_imgs = {}
    if rerun is not None:
        if os.path.isdir(rerun):
            rerun_files = {
                k: os.path.join(rerun, sub_dir, prefix + io_generator.get_name(output_keys[k]))
                for k in rerun_keys
            }

            if not all([os.path.isfile(rerun_file) for rerun_file in rerun_files.values()]):
                raise ValueError(
                    "When rerunning, all rerun files must be present: " + str(rerun_keys)
                )

            for key, rerun_file in rerun_files.items():
                if os.path.isfile(rerun_file):
                    try:
                        rerun_imgs[key] = nilearn.image.load_img(rerun_file).get_fdata()
                    except Exception:
                        print("Error loading rerun imgs. Imgs will be recomputed.")
                        rerun_imgs.clear()
                        break
    return rerun_imgs


def get_io_generator(ref_img, convention, out_dir, prefix, verbose):
    io_generator = tedana.io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        verbose=verbose,
    )
    return io_generator


def save_maps(img_maps, img_keys, io_generator):
    for img_map, img_key in zip(img_maps, img_keys, strict=True):
        output_key = config_sage.get_output_keys()[img_key]
        if output_key is None:
            raise ValueError("invalid output key")
        elif img_map is not None:
            io_generator.save_file(img_map, output_key)


def gen_sub_dirs(sub_dirs):
    if not isinstance(sub_dirs, list):
        sub_dirs = [sub_dirs]
    nested_dir = None
    for sub_dir in sub_dirs:
        if sub_dir is not None:
            if nested_dir is None:
                nested_dir = os.path.abspath(sub_dir)
            else:
                nested_dir = os.path.join(nested_dir, sub_dir)
            if not os.path.isdir(nested_dir):
                os.mkdir(nested_dir)
    if nested_dir is None:
        raise ValueError("invalid subdirectories")
    return nested_dir


""" SCRAP
io_generator.save_file(s0_I_maps, get_output_key("s0I"))
    if s0_II_maps is not None:
        io_generator.save_file(s0_II_maps, get_output_key("s0II"))
    if delta_maps is not None:
        io_generator.save_file(delta_maps, get_output_key("delta"))
    io_generator.save_file(t2star_maps, get_output_key("t2star"))
    io_generator.save_file(t2_maps, get_output_key("t2"))
    if rmspe is not None:
        io_generator.save_file(rmspe, get_output_key("rmspe"))


"""
