function opt = all_HDFS_tumors
% Description: Configuration file for createCSV and train_test_split_HDFS for 
% all labelled tumor images (ICC, HCC, MCRC)

% OUTPUT: opt - struct containing variables defined here

% Environment: MATLAB R2021a
% Author: Katy Scott
% Created: Feb 7, 2022

    % Dimensions for image resize step
    % (32 x 32 is for LeNet)
    % (299 x 299 is Inception requirement)
    % (1024 x 1024 is DeepConvSurv requirement)
    opt.ImageSize = [221 221];
    
%     % Location of image files for tumor image set
    opt.ImageLoc = "../../Data/All_CT/Tumor/";
    
    % Label spreadsheet for original images
    opt.Labels = "../../HDFS/Labels/HCC_MCRC_ICC_HDFS_labels.xlsx";
    
    % Location of bin folder to output tumor image slice set at end of
    % new_preprocessMHA
    opt.BinLoc = strcat("../../Data/Images/Labelled_Tumors/", string(opt.ImageSize(1)), "/Original/");
    
    % Output CSV setup for createCSV
    opt.CSVname = "../../HDFS/Labels/Tumor/HCC_MCRC_ICC_HDFS_labelled_tumor.csv";
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'HDFS_Code', 'HDFS_Time'};
    
    % String to use to find the correct image type in directories
    opt.search_string = "*umor*";
    
    % Test percentage for train-test split
    opt.TestSize = 0.25;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    
    % Output spreadsheets with patient level train and test labels
    opt.TrainLabels = strcat("../../HDFS/Labels/Tumor/HCC_MCRC_ICC_HDFS_tumor_", string(train_perc),"_", string(test_perc),"_train.xlsx");
    opt.TestLabels = strcat("../../HDFS/Labels/Tumor/HCC_MCRC_ICC_HDFS_tumor_", string(train_perc),"_", string(test_perc),"_test.xlsx");
    
    % Directories for train and test bin images
    opt.TrainDestination = strcat("../../Data/Images/Labelled_Tumors/", string(opt.ImageSize(1)), "/HCC_MCRC_ICC_", string(train_perc),"_", string(test_perc),"/train/");
    opt.TestDestination = strcat("../../Data/Images/Labelled_Tumors/", string(opt.ImageSize(1)), "/HCC_MCRC_ICC_", string(train_perc),"_", string(test_perc),"/test/");
     
end