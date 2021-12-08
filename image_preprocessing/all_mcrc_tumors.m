function opt = all_mcrc_tumors
%Description: Configuration file for createCSV for MCRC tumor images
%
%OUTPUT: opt - struct containing variables defined here
%
%Environment: MATLAB R2021a
%Notes: 
%Author: Katy Scott
%Created: Dec 7, 2021
%Updates:
    
    % Dimensions for image resize step in preprocessMHA 
    % (32 x 32 is for LeNet)
    % (299 x 299 is Inception requirement)
    % (1024 x 1024 is DeepConvSurv requirement)
    opt.ImageSize = [32 32];

    % Location of bin folder containing tumour image slice set with zeros
    % in background for use in createCSV
    opt.ZeroLoc = strcat("../../Data/Images/Tumors/", string(opt.ImageSize(1)),"/Zero/");
    % Location of bin folder containing tumour image slice set with NaNs
    % in background for use in createCSV
    opt.NaNLoc = strcat("../../Data/Images/Tumors/", string(opt.ImageSize(1)),"/NaN/");
    
    % Spreadsheet of labels, excel file, for use in createCSV.m
    opt.Labels = "../../Data/TCIA_CRLM_Cases_Final_De-identified.xlsx";
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
    opt.all_file_dir = "../../Data/MCRC/Tumor/";
    opt.destination_dir = "../../Data/MCRC/labelled_only/tumors/";
end