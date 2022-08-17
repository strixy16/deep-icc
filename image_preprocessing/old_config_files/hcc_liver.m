function opt = hcc_liver
%Description: Configuration file for HCC liver images
%
%OUTPUT: opt - struct containing variables defined here
%
%Environment: MATLAB R2021a
%Notes: 
%Author: Katy Scott
%Created: Feb 15, 2022

    % Spreadsheet of labels, excel file
    opt.Labels = "/home/katy/Data/ICC/HDFS/HCC_HDFS_survival.xlsx";
    
    % Location of all mhd and raw image files
    opt.all_file_dir = "../../Data/HCC/liver/";
    % Destination to copy labelled images to in findLabelled
    opt.destination_dir = "../../Data/All_CT/Liver/";
    
    % String to use to find the correct image type in directories
    opt.search_string = "*iver*";
end