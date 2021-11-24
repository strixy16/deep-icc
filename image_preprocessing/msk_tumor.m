function opt = msk_tumor
%Description: Configuration file for preprocessMHA for MSK tumor
%             image set
%
%OUTPUT: opt - struct containing variables defined here
%
%Environment: MATLAB R2020b
%Notes: 
%Author: Katy Scott
%Created: 14 Jan 2021
%Updates:
% Feb 16, 2021 - updated comments, added OutputCSV for createCSV
% Feb 19, 2021 - updated BinLoc so all tumor images go in one place
%              - added ImageSize argument
    
    % Dimensions for image resize step in preprocessMHA 
    % (32 x 32 is for LeNet)
    % (299 x 299 is Inception requirement)
    % (1024 x 1024 is DeepConvSurv requirement)
    opt.ImageSize = [32 32];

    % Location of image files for tumour image set, for use in preprocessMHA
    opt.ImageLoc = "../../Data/cholangio/MSK/tumor/";
    
    % Location of bin folder containing tumour image slice set, for use in
    % preprocess MHA and createCSV
    % NOTE: this has to have subfolders NaN and zero in it 
    opt.BinLoc = strcat("../../Data/Images/Tumors/", string(opt.ImageSize(1)), "/");
    
    
    % Spreadsheet of labels, excel file, for use in createCSV.m
    opt.Labels = "../../Data/RFS_Scout.xlsx";
    % File name + location to output in createCSV.m
    opt.OutputCSV = "../../Data/Labels/MSK_RFS.csv";
end