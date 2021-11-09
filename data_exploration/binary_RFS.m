% Load patient level RFS label data for median and mean calculation
rfs_pat = readtable("../../Data/RFS_Scout.xlsx");

pat_median = median(rfs_pat.RFS);
pat_mean = mean(rfs_pat.RFS);


% Load slice level RFS label data to apply binarization to

rfs_slice = readtable("../../Data/Labels/RFS_all_tumors_zero.csv");

slice_median = median(rfs_slice.RFS_Time);
slice_mean = mean(rfs_slice.RFS_Time);

fprintf('Median: patient = %.0f   slice = %.0f\n', pat_median, slice_median);
fprintf('Mean:   patient = %f   slice = %.4f\n', pat_mean, slice_mean);

% Long survival = 1, short survival = 0
surv_pat = rfs_pat.RFS > pat_median;

surv_slice = rfs_slice.RFS_Time > pat_median;

longsurv_pat_count = sum(surv_pat);

longsurv_slice_count = sum(surv_slice);

binrfs_slice = rfs_slice;

binrfs_slice.RFS_Binary = surv_slice;

writetable(binrfs_slice, "../../Data/Labels/bin_RFS_all_tumors_zero.csv", 'writevariablenames', 1);