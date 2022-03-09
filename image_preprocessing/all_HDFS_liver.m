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
    
    % Location of image files for tumor image set
%     opt.ImageLoc = "../../Data/All_CT/Liver/";
    
    % Location of bin folder to output tumor image slice set at end of
    % new_preprocessMHA
    opt.BinLoc = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Images/Labelled_Liver/", string(opt.ImageSize(1)), "/Original/");
    
    % Output CSV setup for createCSV
    opt.CSVname = "/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Liver/HCC_MCRC_ICC_HDFS_labelled_liver.csv";
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'HDFS_Code', 'HDFS_Time'};
    
    opt.Labels = "/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Liver/HCC_MCRC_ICC_HDFS_liver_labels.xlsx";
    
    % String to use to find the correct image type in directories
    opt.search_string = "*iver*";
    
    % Test percentage for train-test split
    opt.TestSize = 0.20;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    % Output spreadsheets with patient level train and test labels
    opt.TrainLabels = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Liver/HCC_MCRC_ICC_HDFS_liver_", string(train_perc),"_", string(test_perc),"_train.xlsx");
    opt.TestLabels = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Liver/HCC_MCRC_ICC_HDFS_liver_", string(train_perc),"_", string(test_perc),"_test.xlsx");
    
    % Directories for train and test bin images
    opt.TrainDestination = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Images/Labelled_Liver/412/HCC_MCRC_ICC_", string(train_perc),"_", string(test_perc),"/train/");
    opt.TestDestination = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Images/Labelled_Liver/412/HCC_MCRC_ICC_", string(train_perc),"_", string(test_perc),"/test/");
     
end