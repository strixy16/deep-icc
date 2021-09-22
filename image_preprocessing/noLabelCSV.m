function noLabelCSV(conf_f, background)
% Name: noLabelCSV
% Description: Function to generate a CSV file to correspond patients with
% images and slices from preprocessMHA. These patients are unlabelled.
%
% INPUT:
%   conf_f       -- configuration file for different datasets
%   background   -- either "zeros" or "nans"
%
% OUTPUT:
%   A CSV file containing slice file names, associated patient number, and
%   slice number
%
% Environment: MATLAB R2021a
% Author: Katy Scott

    % Getting variables from configuration file
    if ischar(conf_f)
        conf_f = str2func(conf_f);
        options = conf_f();
    else
        options = conf_f;
    end

    if background == "nans"
        % use location of bin files with nans in background
        bin_dir = options.NaNLoc;
        output_fname = options.NoLabelNaNCSV;
        % Making sure output directory exists, create it if it doesn't
        nan_dir = fileparts(output_fname);
        if ~exist(nan_dir, 'dir')
            mkdir(nan_dir);
        end
    else
        % use location of bin files with zeros in background
        if background ~= "zeros"
            disp("Incorrect input for background, using zeros.")
        end
        bin_dir = options.ZeroLoc;
        output_fname = options.NoLabelZeroCSV;
        % Making sure output directory exists, create it if it doesn't
        zero_dir = fileparts(output_fname);
        if ~exist(zero_dir, 'dir')
            mkdir(zero_dir);
        end
    end
    
    % Get list of all bin files
    bin_files = dir(fullfile(bin_dir, '*.bin'));
    % Have folder as structure, change to table for stuff later on
    imgfiles_allinfo = struct2table(bin_files);
    % Extract file names and sort names alphanumerically, now a cell array
    imgfilenames = natsort(imgfiles_allinfo{:,'name'});
    
    % Load in patient label data as a table
    img_labels = readtable(options.Labels);
    % Add an underscore to the end of the ScoutIDs so that a file name that
    % ends in 5 is different from one that ends in 50 for substring use
    mod_patientnames = strcat(img_labels.ScoutID, '_');

    % Find indices of slices from labelled patients
    labelled_idx = contains(imgfilenames, mod_patientnames);
    % Filter out slices of patients without labels
    nolabel_imgfilenames = imgfilenames(labelled_idx==0);
    
    % Get list of patients without labels
%     short_imgfilenames = cell(size(nolabel_imgfilenames, 1), 1);
%     
%     for ind=1:size(nolabel_imgfilenames,1)
%         uidx = strfind(nolabel_imgfilenames{ind},'_');
%         short_imgfilenames{ind} = nolabel_imgfilenames{ind}(1:uidx(end-1));
%     end
%     nolabel_pats = unique(short_imgfilenames, 'stable');
%     writecell(nolabel_pats, '../../Data/noRFSlabel_patients.xlsx'); 
    
    pat_count = 0;
    slice_count = 0;
    curr_patient = 'Init';
    patient_all_data = cell2table(cell(0,3), 'VariableNames', {'File', 'Pat_ID', 'Slice_Num'});
    
    for file_idx=1:size(nolabel_imgfilenames,1)
        % take 6 characters off the end of the filename for comparison with
        % previous file name to see if it's the same patient
        patient = nolabel_imgfilenames{file_idx}(1:end-6);
       
        if contains(patient, curr_patient)
            % Slice is from the same patient
            slice_count = slice_count + 1;
            
        else
            % Slice is from new patient
            curr_patient = patient;
            pat_count = pat_count + 1;
            slice_count = 1;
        end
        
        patient_all_data = [patient_all_data;
                            nolabel_imgfilenames{file_idx}, ... 
                            num2cell(pat_count), ...
                            num2cell(slice_count)];
    end
    
    writetable(patient_all_data, output_fname, 'writevariablenames', 1);
    
end