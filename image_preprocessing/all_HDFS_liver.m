function opt = all_HDFS_liver
% Description: Configuration file for new_preprocessMHA and createCSV for 
% all labelled liver images (ICC, HCC, MCRC)

% OUTPUT: opt - struct containing variables defined here

% Environment: MATLAB R2021a
% Author: Katy Scott
% Created:Feb 15, 2022

    % Dimensions for image resize step
    % TODO: need to find out what the default size the livers are
    opt.ImageSize = [412 412];
    
    % Location of image files for liver image set
    opt.ImageLoc = "../../Data/All_CT/Liver/";
    
    % Spreadsheet with labels for liver image set
    opt.Labels = "../../HDFS/Labels/HCC_MCRC_ICC_HDFS_labels.xlsx";
    
    % Location of bin folder to output liver image slice set at end of
    % new_preprocessMHA
    opt.BinLoc = strcat("../../Data/Images/Labelled_Liver/", string(opt.ImageSize(1)), "/Original/");
    
    % Output CSV setup for createCSV
    opt.CSVname = "../../HDFS/Labels/Liver/HCC_MCRC_ICC_HDFS_labelled_liver.csv";
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'HDFS_Code', 'HDFS_Time'};
    
    % String to use to find the correct image type in directories
    opt.search_string = "*iver*";
    
    % Test percentage for train-test split
    opt.TestSize = 0.25;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    % Output spreadsheets with patient level train and test labels
    opt.TrainLabels = strcat("../../HDFS/Labels/Liver/HCC_MCRC_ICC_HDFS_liver_", string(train_perc),"_", string(test_perc),"_train.xlsx");
    opt.TestLabels = strcat("../../HDFS/Labels/Liver/HCC_MCRC_ICC_HDFS_liver_", string(train_perc),"_", string(test_perc),"_test.xlsx");
    
    % Directories for train and test bin images
    opt.TrainDestination = strcat("../../Data/Images/Labelled_Liver/", string(opt.ImageSize(1)), "/HCC_MCRC_ICC_", string(train_perc),"_", string(test_perc),"/train/");
    opt.TestDestination = strcat("../../Data/Images/Labelled_Liver/", string(opt.ImageSize(1)), "/HCC_MCRC_ICC_", string(train_perc),"_", string(test_perc),"/test/");
     
end