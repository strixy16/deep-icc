function opt = all_tumors
%Description: Configuration file for createCSV for both MSK and Erasmus
%ICC tumor images
%
%OUTPUT: opt - struct containing variables defined here
%
%Environment: MATLAB R2021a
%Notes: 
%Author: Katy Scott
%Created: Dec 6, 2021
%Updates:
    
    msk_options = msk_tumor;
    erasmus_options = erasmus_tumors;

    opt.ImageSize = msk_options.ImageSize;

    % Location of bin folder containing tumour image slice set with zeros
    % in background for use in createCSV
    opt.ZeroLoc = strcat("../../Data/Images/Tumors/", string(opt.ImageSize(1)),"/Zero/");
    % Location of bin folder containing tumour image slice set with NaNs
    % in background for use in createCSV
    opt.NaNLoc = strcat("../../Data/Images/Tumors/", string(opt.ImageSize(1)),"/NaN/");
    
    % Spreadsheet of labels, excel file, for use in createCSV.m
    opt.Labels = "../../Data/RFS_Scout.xlsx";
    % Header
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'RFS_Code', 'RFS_Time'};
    
%     opt.Label1 = 'RFS Code'; % if a patient had cancer recurrence or not
%     opt.Label2 = 'RFS Time'; % Time to recurrence
    % File name + location to output in createCSV.m
    opt.ZeroCSV = strcat("../../Data/Labels/", "/RFS_all_tumors_zero.csv");
    opt.NaNCSV = strcat("../../Data/Labels/", "/RFS_all_tumors_NaN.csv");
    
    opt.GeneZeroCSV = strcat("../../Data/Labels/", "/RFS_gene_tumors_zero.csv");
    opt.GeneNaNCSV = strcat("../../Data/Labels/", "/RFS_gene_tumors_NaN.csv");
    
    opt.NoLabelZeroCSV = strcat("../../Data/Labels/", "/nl_all_tumors_zero.csv");
    opt.NoLabelNaNCSV = strcat("../../Data/Labels/", "/nl_all_tumors_NaN.csv");
    
    % Variables for findLabelled.m
    opt.all_file_dir = "../../Data/cholangio/AllSources/Tumor/";
    opt.labels_file = "../../Data/RFS_Scout.xlsx";
    opt.destination_dir = "../../Data/cholangio/labelled_only/tumors/";
end