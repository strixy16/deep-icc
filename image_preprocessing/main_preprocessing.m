% Main script for image preprocessing

% Set which config file to use in preprocessing
config = all_HDFS_liver;

% Crop and save images as bin files
new_preprocessMHA(config);

% Create label spreadsheet for bin files
createCSV_HDFS(config)

