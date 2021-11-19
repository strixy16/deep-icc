load('imagecounts.mat');

tumor_names = all_tumor_files.name;
liver_names = all_liver_files.name;
vol_names = all_vol_files.name;


%  = tumor_names{:,1}(1:end-9);

% removing Tumor.mhd, liver_segmented.mhd, and Tomogram.mhd from file names
% for comparison
tumor_pat_names = cellfun(@(x) x(1:end-9), tumor_names, 'UniformOutput', false);
liver_pat_names = cellfun(@(x) x(1:end-19), liver_names, 'UniformOutput', false);
vol_pat_names = cellfun(@(x) x(1:end-12), vol_names, 'UniformOutput', false);


missing_liver = setdiff(tumor_pat_names, liver_pat_names);
missing_tumor = setdiff(liver_pat_names, tumor_pat_names);

missing_volume = setdiff(tumor_pat_names, vol_pat_names);
missing_tumor2 = setdiff(vol_pat_names, tumor_pat_names);