import SimpleITK as sitk
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import six

import radiomics
from radiomics import firstorder, imageoperations

""" Following https://github.com/AIM-Harvard/pyradiomics/blob/master/notebooks/helloFeatureClass.ipynb """

# Get name of all tumor files to load
tumor_dir = "/media/katy/Data/ICC/Data/All_CT/Tumor"
all_tumor_files = os.listdir(tumor_dir)
tumor_files_mhd = [x for x in all_tumor_files if "mhd" in x]
# Remove tumors that don't have corresponding liver image (confirmed externally)
tumor_files_mhd.remove('005_ICCrecurrence_Tumor.mhd')
tumor_files_mhd.remove('057_ICCrecurrence_Tumor.mhd')
# Sort files so they align with liver files
tumor_files_mhd.sort()

# Get name of all liver files to load
liver_dir = "/media/katy/Data/ICC/Data/All_CT/Liver"
all_liver_files = os.listdir(liver_dir)
liver_files_mhd = [y for y in all_liver_files if "mhd" in y]
# Sort files so they align with tumor files
liver_files_mhd.sort()

# Confirming same number of tumor and liver files
print("Tumor file count: ", len(tumor_files_mhd))
print("Liver file count: ", len(liver_files_mhd))

cancer_type = "HCC_MCRC_ICC"
lbl_file_name = "/media/katy/Data/ICC/HDFS/" + cancer_type + "_HDFS_liver_labels.xlsx"
liver_labels = pd.read_excel(lbl_file_name)

# Dataframe to store features for each patient
liver_features_df = pd.DataFrame(columns=['ScoutID', 'Entropy', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])
tumor_features_df = pd.DataFrame(columns=['ScoutID', 'Entropy', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])
liver_features_df['ScoutID'] = liver_labels['ScoutID']
tumor_features_df['ScoutID'] = liver_labels['ScoutID']

for idx in range(len(liver_labels)):
    scoutid = liver_features_df['ScoutID'][idx]
    scout_id = scoutid + "_"
    print("Patient ", idx,": ", scout_id)
    ## DATA LOADING ##

    tumor_file = [f for f in tumor_files_mhd if scout_id in f]
    liver_file = [f for f in liver_files_mhd if scout_id in f]
    if len(tumor_file) != 1:
        print(tumor_file)
        raise Exception("scout_id should find 1 tumor file. Caught %i files.".format(len(tumor_file)))
    if len(liver_file) != 1:
        print(liver_file)
        raise Exception("scout_id should find 1 liver file. Caught %i files.".format(len(tumor_file)))

        
    # Loading patient idx image
    tumor_file = tumor_file[0]
    liver_file = liver_file[0]

    # print("Tumor file: ", tumor_file)
    # print("Liver file: ", liver_file)

    # Load tumor image
    tumor_img_path = os.path.join(tumor_dir, tumor_file)
    tumor_mhd_image = sitk.ReadImage(tumor_img_path, imageIO="MetaImageIO", outputPixelType=sitk.sitkInt64)
    # Convert to array for mask making
    tumor_arr = sitk.GetArrayFromImage(tumor_mhd_image)

    liver_img_path = os.path.join(liver_dir, liver_file)
    liver_mhd_image = sitk.ReadImage(liver_img_path, imageIO="MetaImageIO", outputPixelType=sitk.sitkInt64)
    # Convert to array for mask making
    liver_arr = sitk.GetArrayFromImage(liver_mhd_image)

    # This one patient has too many slices and causes the program to run out of memory
    # So I select from 700 down to the 300 that have liver/tumor pixels with some boundary slices
    if scout_id == "ICC_Radiogen_Add28_":
        del tumor_mhd_image, liver_mhd_image
        tumor_arr = tumor_arr[300:600,:,:]
        liver_arr = liver_arr[300:600,:,:]

    ## MASK MAKING ##
    # Make masks for tumor and liver
    # -1000 is background value in these images
    tumor_mask = tumor_arr != -1000
    liver_mask = liver_arr != -1000
    # Convert to numeric values for easy handling
    tumor_mask = tumor_mask.astype(float)
    liver_mask = liver_mask.astype(float)

    # Remove tumor from liver image and mask
    # Set tumor to 0s in tumor mask and remove from liver mask
    notumor_livermask = liver_mask * (1 - tumor_mask)
    # Set tumor to 0s in tumor mask, use to set to 0 in liver image, add -1000 to tumor to set as background
    notumor_liver = liver_arr * (1 - tumor_mask) + (tumor_mask * -1000)

    # Convert no-tumor liver image and mask back to sitk.Image type for feature extraction
    liver_image = sitk.GetImageFromArray(notumor_liver)
    liver_mask = sitk.GetImageFromArray(notumor_livermask)
    # Set pixels to int type for pyradiomics functions
    liver_image = sitk.Cast(liver_image, sitk.sitkInt64)
    liver_mask = sitk.Cast(liver_mask, sitk.sitkInt64)

    tumor_image = sitk.GetImageFromArray(tumor_arr)
    tumor_mask = sitk.GetImageFromArray(tumor_mask)
    tumor_image = sitk.Cast(tumor_image, sitk.sitkInt64)
    tumor_mask = sitk.Cast(tumor_mask, sitk.sitkInt64)

    ## IMAGE PREPROCESSING ##
    # Extraction settings 
    settings = {}
    settings['resampledPixelSpacing'] = [1, 1, 1]
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True

    # Resampling image
    # print("Before resampling: ", liver_image.GetSize())

    interpolator = settings.get('interpolator')
    # Removes slices with no/few image pixels
    resampledPixelSpacing = settings.get('resampledPixelSpacing')
    if interpolator is not None and resampledPixelSpacing is not None:
        liver_image, liver_mask = imageoperations.resampleImage(liver_image, liver_mask, **settings)
        tumor_image, tumor_mask = imageoperations.resampleImage(tumor_image, tumor_mask, **settings)

    # print("After resampling: ", liver_image.GetSize())

    # Crop image to mask
    # bb is bounding box, upon which image and mask are cropped
    liver_bb, liver_correctedMask = imageoperations.checkMask(liver_image, liver_mask, label=1)
    if liver_correctedMask is not None:
        liver_mask = liver_correctedMask

    tumor_bb, tumor_correctedMask = imageoperations.checkMask(tumor_image, tumor_mask, label=1)
    if tumor_correctedMask is not None:
        tumor_mask = tumor_correctedMask

    liver_croppedImage, liver_croppedMask = imageoperations.cropToTumorMask(liver_image, liver_mask, liver_bb)
    tumor_croppedImage, tumor_croppedMask = imageoperations.cropToTumorMask(tumor_image, tumor_mask, tumor_bb)

    # Calculate First Order features
    liver_fof = firstorder.RadiomicsFirstOrder(liver_croppedImage, liver_croppedMask, **settings)
    liver_fof.enableFeatureByName('Entropy', True)
    liver_fof.enableFeatureByName('Mean', True)
    liver_fof.enableFeatureByName('Variance', True)
    liver_fof.enableFeatureByName('Skewness', True)
    liver_fof.enableFeatureByName('Kurtosis', True)

    tumor_fof = firstorder.RadiomicsFirstOrder(tumor_croppedImage, tumor_croppedMask, **settings)
    tumor_fof.enableFeatureByName('Entropy', True)
    tumor_fof.enableFeatureByName('Mean', True)
    tumor_fof.enableFeatureByName('Variance', True)
    tumor_fof.enableFeatureByName('Skewness', True)
    tumor_fof.enableFeatureByName('Kurtosis', True)

    liver_result = liver_fof.execute()
    tumor_result = tumor_fof.execute()

    for (key, val) in six.iteritems(liver_result):
        # print('    ', key, ':', val)
        # Add features to dataframe
        liver_features_df[key][idx] = float(val)
    
    for (key, val) in six.iteritems(tumor_result):
        tumor_features_df[key][idx] = float(val)

    # del tumor_image, tumor_arr, tumor_mask, tumor_mhd_i, tumor_croppedMask, tumor_croppedImage, tumor_bb
    # del liver_image, liver_arr, liver_mask, liver_mhd_image, liver_croppedMask, liver_croppedImage, liver_bb
    # del notumor_liver, notumor_livermask, liver_result, tumor_result, liver_fof, tumor_fof,
# END patient loop

liver_feature_fname = "/media/katy/Data/ICC/HDFS/" + cancer_type + "_HDFS_liver_firstorderfeatures.xlsx"
liver_features_df.to_excel(liver_feature_fname, index=False)

tumor_feature_fname = "/media/katy/Data/ICC/HDFS/" + cancer_type + "_HDFS_tumor_firstorderfeatures.xlsx"
tumor_features_df.to_excel(tumor_feature_fname, index=False)

print("Feature selection complete")
