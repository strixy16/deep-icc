function opt = HDFS_train_liver
% Description: Configuration file for main_train_test_script for all 
% labelled liver images (ICC, HCC, MCRC)

% OUTPUT: opt - struct containing variables defined here

% Environment: MATLAB R2021a
% Author: Katy Scott

    % Get train test split percentage from parent config file
    HDFS_liver_options = all_HDFS_liver;
    opt.TestSize = HDFS_liver_options.TestSize;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    % Location for training bin files
    opt.BinLoc = HDFS_liver_options.TrainDestination;
    
    % Output CSV setup for createCSV
    opt.CSVname = strcat("../../HDFS/Labels/Liver/HCC_MCRC_ICC_HDFS_", string(train_perc),"_", string(test_perc),"_train_liver.csv");
    opt.CSV_header = HDFS_liver_options.CSV_header;
    
    % Output spreadsheets with patient level train labels
    opt.Labels = HDFS_liver_options.TrainLabels;
end