train_files = readcell("../../Data/train_slice_files.xlsx");
test_files = readcell("../../Data/test_slice_files.xlsx");

% Remove header line 
train_files = train_files(2:end);
test_files = test_files(2:end);

train_pats = cell(size(train_files,1),1);
test_pats = cell(size(test_files,1),1);

for idx = 1:size(train_files,1)
    und_idx = strfind(train_files{idx}, '_');
    trunc_file_name = train_files{idx}(1:und_idx(end-2)-1);
    train_pats{idx,:} = trunc_file_name;
end

for idx = 1:size(test_files,1)
    und_idx = strfind(test_files{idx}, '_');
    trunc_file_name = test_files{idx}(1:und_idx(end-2)-1);
    test_pats{idx,:} = trunc_file_name;
end

unique_train_pats = strip(unique(train_pats));
unique_test_pats = strip(unique(test_pats));

train_tbl = cell2table([{'ScoutID'};unique_train_pats]);
test_tbl = cell2table([{'ScoutID'};unique_test_pats]);

writetable(train_tbl, "../../Data/train_pats.xlsx", 'WriteVariableNames', false);
writetable(test_tbl, "../../Data/test_pats.xlsx", 'WriteVariableNames', false);



