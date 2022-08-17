function opt = HDFS_train_tumors
% Description: Configuration file for main_train_test_script for all 
% labelled tumour images (ICC, HCC, MCRC)
% OUTPUT: opt - struct containing variables defined here

% Environment: MATLAB R2021a
% Author: Katy Scott

    % Get train test split percentage from parent config file
    HDFS_tumor_options = all_HDFS_tumors;
    opt.TestSize = HDFS_tumor_options.TestSize;
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    % Location for testing bin files
    opt.BinLoc = HDFS_tumor_options.TrainDestination;
    
    % Output CSV setup for createCSV
    opt.CSVname = strcat("../../HDFS/Labels/Tumor/HCC_MCRC_ICC_HDFS_", string(train_perc),"_", string(test_perc),"_train_tumors.csv");
    opt.CSV_header = HDFS_tumor_options.CSV_header;
    
    % Output spreadsheets with patient level train labels
    opt.Labels = HDFS_tumor_options.TrainLabels;
end