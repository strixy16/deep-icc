MSK_file = "../Data/TextureFeatures16D1.xlsx";
eras_file = "../Data/TextureFeaturesErasmus16D1.xlsx";

tbl_MSK = readtable(MSK_file);
tbl_eras = readtable(eras_file);

tbl_allfeat = [tbl_MSK; tbl_eras];

tbl_RFS = readtable("../Data/RFS_Scout.xlsx");

% Need to get out the ScoutID up to the second underscore
% Removing _Tumor.mhd

% Initializing column to add to tbl_allfeat
ScoutID = cell(length(tbl_allfeat.Var1),1);

for i=1:length(tbl_allfeat.Var1)
    % finding index of underscores
    uidx = strfind(tbl_allfeat.Var1{i},'_');
    % Getting last index of file name without _Tumor.mhd
    nidx = uidx(end);
    % storing ID for new column
    ScoutID{i} = tbl_allfeat.Var1{i}(2:nidx-1);
    
end
% Inserting new ID column and removing old one
tbl_allfeat = addvars(tbl_allfeat, ScoutID, 'Before', 'GLCM1');
tbl_allfeat = tbl_allfeat(:, 2:end);

% Adding RFS labels to the table and getting only patients with RFS label
tbl_featandlabels = innerjoin(tbl_RFS, tbl_allfeat);

% Feature selection 
% Get features and labels out separately
X = tbl_featandlabels(:,4:end);
Y = tbl_featandlabels.RFS;

[idx, scores] = fscmrmr(X, Y);
figure
bar(scores(idx(1:25)))
xlabel('Predictor rank')
ylabel('Predictor importance score')
title("MRMR Feature Selection for CT Texture Features")
top10 = [tbl_featandlabels(:,1:3) tbl_featandlabels(:,idx(1:10)+3)];
writetable(top10, '../Data/selectTextFeatures.xlsx');

Mdl = fitrensemble(X, Y, 'Method', 'bag');

