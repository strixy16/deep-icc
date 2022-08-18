

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



* ***RandomSurvivalForest/***
  * ***hdfs_random_forest.ipynb*** :arrow_right: Jupyter notebook creating Random Survival Forests for the liver, tumour, and liver with tumour images. Uno's c-index calculations and confidence intervals are both calculated within this notebook.

* ***old_code/***: :arrow_right: Code written in process of development. Was either broken and abandoned or absorbed into the files listed above.


## Requirements 📋
To run image preprocessing, MATLAB R2021a or later is required.

To setup the model in a conda environment, run the following:

```setup
conda create --name deepicc -f environment.yml
jupyter lab
```
Make sure the kernel is set to Python3 and you can now run the main and data_explore notebooks.

## Image Preprocessing 🖼️

To use the MATLAB preprocessing functions, preprocessMHA and createCSV, you will need:
* A directory of corresponding MHD and raw files
* A spreadsheet with labels for each sample

First create a configuration file for your dataset. You can follow the setup of msk_tumor and all_tumors. The variables required are:

*For preprocessMHA*
* ImageSize: dimensions to crop your image to for the network (ex. \[256 x 256]) 
* ImageLoc: path to the directory your MHD and raw files are in
* BinLoc: path to the directory to store the output BIN files

*For createCSV*
* ZeroLoc: path to directory of BIN files with zeros for background
* Labels: path to spreadsheet file containing labels 
* CSV_header: Set headings for label output CSV
* OutputCSV: path and name of output CSV linking labels to BIN files

To crop images based on max height and width for that set, run this command in MATLAB:
``` 
preprocessMHA(config_file);
createCSV(config_file); 
```
This should generate a directory of individual BIN files for each slice of the MHD volume and a corresponding CSV label file.

## Model Training 🏃

Model creation and training is available in `main.ipynb`

The data used to train the KT6 model described in the report is not publicly available. 
Output from training this model is included in the notebook. `ckpts-kt-100` contains the stored training checkpoints that can be loaded into the Tensorboard at the end of the notebook.







