function opt = icc_liver
%Description: Configuration file for ICC liver images
%
%OUTPUT: opt - struct containing variables defined here
%
%Environment: MATLAB R2021a
%Notes: 
%Author: Katy Scott
%Created: Feb 15, 2022

    % Spreadsheet of labels, excel file
    opt.Labels = "/home/katy/Data/ICC/HDFS/RFS_Scout.xlsx";
    
    % Location of all mhd and raw image files
    opt.all_file_dir = "/home/katy/Data/CISC499/Images/ICC/liver/";
    % Destination to copy labelled images to in findLabelled
    opt.destination_dir = "../../Data/All_CT/Liver/";
    
    % String to use to find the correct image type in directories
    opt.search_string = "*iver*";
end