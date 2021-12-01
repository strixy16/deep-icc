function opt = labelled_tumors
%Description: Configuration file for preprocessMHA for labelled tumor
%             image set
%
%OUTPUT: opt - struct containing variables defined and described below
%
%Environment: MATLAB R2021a
%Notes: 
%Author: Katy Scott
%Created: 1 Dec 2021
%Updates:


    % Dimensions for image resize step in preprocessMHA 
    % (32 x 32 is for LeNet)
    % (299 x 299 is Inception requirement)
    % (1024 x 1024 is DeepConvSurv requirement)
    opt.ImageSize = [32 32];
    
    % Location of image files for tumour image set, for use in preprocessMHA
    opt.ImageLoc = "../../Data/cholangio/labelled_only/tumors/";
    
    % Location of bin folder containing tumour image slice set, for use in
    % preprocess MHA and createCSV
    % NOTE: this has subfolders NaN and zero in it 
    opt.BinLoc = strcat("../../Data/Images/Labelled_Tumors/", string(opt.ImageSize(1)), "/");
    
    % Location of bin folder containing tumour image slice set with zeros
    % in background for use in createCSV
    opt.ZeroLoc = strcat("../../Data/Images/Labelled_Tumors/", string(opt.ImageSize(1)),"/Zero/");
    
    % Spreadsheet of labels, excel file, for use in createCSV.m
    opt.Labels = "../../Data/RFS_Scout.xlsx";
    
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'RFS_Code', 'RFS_Time'};
    
    % File name + location to output in createCSV.m
    opt.ZeroCSV = "../../Data/Labels/RFS_labelled_tumors_zero.csv";
end