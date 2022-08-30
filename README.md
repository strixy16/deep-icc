

# Survival Time Prediction with Radiographic Images for Primary and Metastatic Liver Cancers

*Katy Scott* 

This repository is the official implementation of my M.Sc. thesis project.

Thesis document describing experiments and results available here: https://qspace.library.queensu.ca/handle/1974/30225


## File Descriptions :open_file_folder:
* ***image_preprocessing/*** :arrow_right: contains MATLAB code for preprocessing .raw and .mhd files to .bin files for use in main. Configuration files must be generated like all_HDFS_liver.m, HDFS_train_liver.m, and HDFS_test_liver.m to be used in main_preprocessing and main_train_test_script.
  * ***main_preprocessing.m*** :arrow_right: Set config as the main configuration file you want to use, should be set up like all_HDFS_liver.m for example. Crops and saves the corresponding images as bin files and creates label spreadsheet for deep learning code.
  * ***new_preprocessMHA.m*** :arrow_right: Function that converts .raw and .mhd to .bin files. Called in main_preprocessing.
  * ***createCSV_HDFS.m*** :arrow_right: Function that creates a CSV file connecting patients to file names, slices, and HDFS labels. Called in main_preprocessing and main_train_test_script.
  
  * ***main_train_test_script.m*** :arrow_right: Splits data into train and test sets. Main config is same as main_preprocessing. 
  * ***train_test_split_HDFS.m*** :arrow_right: Copies bin files generated by main_preprocessing into train and test folders. Called in main_train_test_script.

Remaining functions in this directory are helper functions or configuration files used in the preprocessing pipeline.

* ***data_exploration/*** :arrow_right: contains Python and MATLAB code for generation of survival curves and data distribution, radiomic feature extraction and feature selection
  * ***ct_metadata.m*** :arrow_right: MATLAB script to get the element spacing from the CT metadata for the images.
  * ***explore_genetics.ipynb*** :arrow_right: Looked at KM survival curves for binary genetic mutation status data and tried using deepsurv network on the data for RFS prediction.
  * ***feature_selection.ipynb*** :arrow_right: Notebook with some of the radiomic feature extraction/selection code. Can be used to generate images of the masked images and liver with tumour removed images.
  * ***feature_selection.py*** :arrow_right: Python script for radiomic feature extraction/selection. Currently set up to extract First Order, NGTDM, and Shape 3D features from the liver with no tumour and tumour images. 
  * ***linear_CPH.ipynb*** :arrow_right: Notebook to perform the assumptions checking for the lienar CPH (censoring and proportional hazards)
  * ***survcurv_cancertype.ipynb*** :arrow_right: Notebook for generation of survival curves and data distribution for each cancer type.

* ***R_code/***
  * ***linear_CPH/*** :arrow_right: code to run a linear Cox Proportional Hazards model using extracted radiomic features. The file name indicates what image set was used and the train-test split (e.g. liver_CPH_90_10 used features from the liver images with no tumour and 90:10 train test split)
  * ***CNN_UnoC_calculations/*** :arrow_right: code to calculate Uno's C-statistic for the results of the deep learning CNN model

* ***RandomSurvivalForest/***
  * ***hdfs_random_forest.ipynb*** :arrow_right: Jupyter notebook creating Random Survival Forests for the liver, tumour, and liver with tumour images. Uno's c-index calculations and confidence intervals are both calculated within this notebook.

* ***old_code/***: :arrow_right: Code written in process of development. Was either broken and abandoned or absorbed into the files listed above.

* ***Main directory*** :arrow_right: Mostly code for CNN model
  * ***hdfs_config.py/hdfs_liver_config.py*** :arrow_right: Configuration files for the CNN model. Use this to set location of data files, model training setup and hyperparameters.
  * ***hdfs_data_loading.py*** :arrow_right: Contains HDFS data class (works for Liver as well)
  * ***hdfs_models.py*** :arrow_right: CNN model architectures and evaluation functions and classes.
  * ***hdfs_train.py*** :arrow_right: Main model training file.
  * ***model_performance_stats.ipynb*** :arrow_right: Notebook used to calculate c-index for each cancer type in the dataset for all models
  * ***requirements.txt*** :arrow_right: Requirements to run the deep learning model (Note: this might be incomplete)


## Requirements 📋
Image preprocessing is completed in MATLAB using R2021A or later.

Linear CoxPH model and Uno C-statistic calculation is completed in R.

All remaining code is completed in Python.


## Radiomic Feature Extraction  🩻



## Image Preprocessing for CNN 🖼️

To use the MATLAB preprocessing functions, you will need:
* A directory of corresponding MHD and raw files
* A spreadsheet with labels for each sample (event, time, cancer type (numbered))

1. First create the configuration files for your dataset. You can follow the setup of all_HDFS_liver.m for main_preprocessing and then add HDFS_train_liver.m, and HDFS_test_liver.m for main_train_test_script.m.
2. Set up the config variable in main_preprocessing.m and main_train_test_script.m
3. Run main_preprocessing.m. This should generate a directory of preprocessed bin files and a spreadsheet of corresponding labels for the CNN model.
4. Run main_train_test_script.m. This should copy the bin files in the corresponding train or test directory and make corresponding labels for them.

## Model Training 🏃

Model creation and training is available in `main.ipynb`

The data used to train the KT6 model described in the report is not publicly available. 
Output from training this model is included in the notebook. `ckpts-kt-100` contains the stored training checkpoints that can be loaded into the Tensorboard at the end of the notebook.







