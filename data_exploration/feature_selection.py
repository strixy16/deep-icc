import SimpleITK as sitk
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import six

import radiomics
from radiomics import firstorder, imageoperations, ngtdm, shape, featureextractor, getFeatureClasses

""" Following https://github.com/AIM-Harvard/pyradiomics/blob/master/notebooks/helloFeatureClass.ipynb """

print(sitk.__version__)
# Get name of all tumor files to load
tumor_dir = "/media/katy/Data/ICC/Data/All_CT/Tumor"
all_tumor_files = os.listdir(tumor_dir)
tumor_files_mhd = [x for x in all_tumor_files if "mhd" in x]
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
lbl_file_name = "/media/katy/Data/ICC/HDFS/Labels/" + cancer_type + "_HDFS_labels.xlsx"
liver_labels = pd.read_excel(lbl_file_name)

# Dataframe to store features for each patient
liver_features_df = pd.DataFrame(columns=['ScoutID'])
tumor_features_df = pd.DataFrame(columns=['ScoutID'])
# liver_features_df['ScoutID'] = liver_labels['ScoutID']
# tumor_features_df['ScoutID'] = liver_labels['ScoutID']

for idx in range(len(liver_labels)):
    scoutid = liver_labels['ScoutID'][idx]
    scout_id = scoutid + "_"
    print("Patient ", idx,": ", scout_id)
    ## DATA LOADING ##

    tumor_files = [f for f in tumor_files_mhd if scout_id in f]
    print(len(tumor_files))
    liver_file = [f for f in liver_files_mhd if scout_id in f]
    # if len(tumor_file) != 1:
    #     print(tumor_file)
    #     raise Exception("scout_id should find 1 tumor file. Caught {:d} files.".format(len(tumor_file)))
    if len(liver_file) != 1:
        print(liver_file)
        raise Exception("scout_id should find 1 liver file. Caught {:d} files.".format(len(liver_file)))

        
    # Loading patient idx image
    # tumor_file = tumor_file[0]
    liver_file = liver_file[0]

    # print("Tumor file: ", tumor_file)
    # print("Liver file: ", liver_file)

    # Load liver first to get size for initializing total tumor mask
    liver_img_path = os.path.join(liver_dir, liver_file)
    liver_mhd_image = sitk.ReadImage(liver_img_path, imageIO="MetaImageIO", outputPixelType=sitk.sitkInt64)
    # Convert to array for mask making
    liver_arr = sitk.GetArrayFromImage(liver_mhd_image)

    # This one patient has too many slices and causes the program to run out of memory
    # So I select from 700 down to the 300 that have liver/tumor pixels with some boundary slices
    if scout_id == "ICC_Radiogen_Add28_":
        del liver_mhd_image
        liver_arr = liver_arr[300:600,:,:]

    # Initialize total tumor mask
    total_tumor_mask = np.full(liver_arr.shape, False, dtype=bool)
    for tfile in tumor_files:
        # Load tumor image
        tumor_img_path = os.path.join(tumor_dir, tfile)
        tumor_mhd_image = sitk.ReadImage(tumor_img_path, imageIO="MetaImageIO", outputPixelType=sitk.sitkInt64)
        # Convert to array for mask making
        tumor_arr = sitk.GetArrayFromImage(tumor_mhd_image)

        # This one patient has too many slices and causes the program to run out of memory
        # So I select from 700 down to the 300 that have liver/tumor pixels with some boundary slices
        if scout_id == "ICC_Radiogen_Add28_":
            del tumor_mhd_image
            tumor_arr = tumor_arr[300:600,:,:]

        # Add this tumor to existing tumor mask
        tumor_mask = tumor_arr != -1000
        total_tumor_mask = np.bitwise_or(total_tumor_mask, tumor_mask)

    ## MASK MAKING ##
    # Make masks for tumor and liver
    # -1000 is background value in these images
    liver_mask = liver_arr != -1000
    # Convert to numeric values for easy handling
    tumor_mask = total_tumor_mask.astype(float)
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
    settings = {'resampledPixelSpacing': [1, 1, 1], 'interpolator': 'sitkBSpline', 'verbose': True}

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
    liver_fof.enableAllFeatures()
    liver_ngtdm = ngtdm.RadiomicsNGTDM(liver_croppedImage, liver_croppedMask, **settings)
    liver_ngtdm.enableAllFeatures()
    liver_shape = shape.RadiomicsShape(liver_croppedImage, liver_croppedMask, **settings)
    liver_shape.enableAllFeatures()

    tumor_fof = firstorder.RadiomicsFirstOrder(tumor_croppedImage, tumor_croppedMask, **settings)
    tumor_fof.enableAllFeatures()
    tumor_ngtdm = ngtdm.RadiomicsNGTDM(tumor_croppedImage, tumor_croppedMask, **settings)
    tumor_ngtdm.enableAllFeatures()
    tumor_shape = shape.RadiomicsShape(tumor_croppedImage, tumor_croppedMask, **settings)
    tumor_shape.enableAllFeatures()

    liver_fof_result = liver_fof.execute()
    liver_ngtdm_result = liver_ngtdm.execute()
    liver_shape_result = liver_shape.execute()
    tumor_fof_result = tumor_fof.execute()
    tumor_ngtdm_result = tumor_ngtdm.execute()
    tumor_shape_result = tumor_shape.execute()

    pd_liver_result = pd.DataFrame([liver_fof_result | liver_ngtdm_result | liver_shape_result])
    pd_tumor_result = pd.DataFrame([tumor_fof_result | tumor_ngtdm_result | tumor_shape_result])

    pd_liver_result.insert(0, 'ScoutID', scoutid)
    pd_tumor_result.insert(0, 'ScoutID', scoutid)

    if idx == 0:
        feature_list = list(pd_liver_result.columns)
        liver_features_df = liver_features_df.reindex(feature_list, axis="columns")
        tumor_features_df = tumor_features_df.reindex(feature_list, axis="columns")
    
    liver_features_df = pd.concat([liver_features_df, pd_liver_result])
    tumor_features_df = pd.concat([tumor_features_df, pd_tumor_result])
# END patient loop

# liver_feature_fname = "/media/katy/Data/ICC/HDFS/FeatureSelection/" + cancer_type + "_HDFS_liver_notumors_features.xlsx"
# liver_features_df.to_excel(liver_feature_fname, index=False)

# tumor_feature_fname = "/media/katy/Data/ICC/HDFS/FeatureSelection/" + cancer_type + "_HDFS_indextumor_features.xlsx"
# tumor_features_df.to_excel(tumor_feature_fname, index=False)

print("Feature selection complete")
