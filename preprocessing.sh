# skull stripping with ROBEX (Iglesias et al., 2011);
# If your system is linux-based, download linux-version ROBEX instead of the all-platforms version 
sh runROBEX.sh [original_file].nii [output_after_skull_stripping].nii

# Bias field correction with N4ITK4 (Tustison et al., 2010)
# pip install SimpleITK before running it
# download code from https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html as N4BiasFieldCorrection.py 
python N4BiasFieldCorrection.py [input_name].nii [output_name].nii





