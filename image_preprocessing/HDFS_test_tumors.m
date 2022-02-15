function opt = HDFS_test_tumors
% Description: Configuration file for new_preprocessMHA and createCSV for 
% all labelled tumor images (ICC, HCC, MCRC)

% OUTPUT: opt - struct containing variables defined here

% Environment: MATLAB R2021a
% Author: Katy Scott
% Created: Dec 13, 2021

    % Dimensions for image resize step
    % (32 x 32 is for LeNet)
    % (299 x 299 is Inception requirement)
    % (1024 x 1024 is DeepConvSurv requirement)
    opt.ImageSize = [220 220];
    
    % Location of image files for tumor image set
%     opt.ImageLoc = "../../Data/All_CT/Tumor/";
    
    % Test percentage for train-test split
    opt.TestSize = 0.25;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    % Location of bin folder to output tumor image slice set at end of
    % new_preprocessMHA
    opt.BinLoc = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Images/Labelled_Tumors/221/HCC_MCRC_ICC_", string(train_perc),"_", string(test_perc),"/test/");
    
    % Output CSV setup for createCSV
    opt.CSVname = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/HCC_MCRC_ICC_HDFS_", string(train_perc),"_", string(test_perc),"_test_tumors.csv");;
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'HDFS_Code', 'HDFS_Time'};
    
    opt.Labels = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/HCC_MCRC_ICC_HDFS_", string(train_perc),"_", string(test_perc),"_test.xlsx");
    
    
end