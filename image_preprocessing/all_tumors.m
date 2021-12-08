function opt = all_tumors
% Description: Configuration file for new_preprocessMHA for all labelled 
% tumor images (ICC, HCC, MCRC)

% OUTPUT: opt - struct containing variables defined here

% Environment: MATLAB R2021a
% Author: Katy Scott
% Created: Dec 7, 2021

    % Dimensions for image resize step
    % (32 x 32 is for LeNet)
    % (299 x 299 is Inception requirement)
    % (1024 x 1024 is DeepConvSurv requirement)
    opt.ImageSize = [32 32];
    
    % Location of image files for tumor image set
    opt.ImageLoc = "../../Data/All_CT/Tumor/";
    
    % Location of bin folder to output tumor image slice set at end of
    % new_preprocessMHA
    opt.BinLoc = strcat("../../Data/Images/Labelled_Tumors/", string(opt.ImageSize(1)), "/");
    

end