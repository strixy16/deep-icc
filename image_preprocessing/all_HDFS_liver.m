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
    opt.ImageLoc = "../../Data/All_CT/Liver/";
    
    % Location of bin folder to output tumor image slice set at end of
    % new_preprocessMHA
    opt.BinLoc = strcat("../../Data/Images/Labelled_Liver/", string(opt.ImageSize(1)), "/Original/");
    
    % Output CSV setup for createCSV
    opt.CSVname = "../../Data/Labels/HCC_MCRC_ICC_HDFS_labelled_liver.csv";
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'HDFS_Code', 'HDFS_Time'};
    
    opt.Labels = "/home/katy/Data/ICC/HDFS/HCC_MCRC_ICC_HDFS_liver_labels.xlsx";
    
    % String to use to find the correct image type in directories
    opt.search_string = "*iver*";
    
end