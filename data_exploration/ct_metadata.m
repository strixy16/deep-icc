dir_path = "/home/katy/Data/ICC/Data/All_CT/Tumor/";

all_mhd_files = dir(strcat(dir_path, "*ICC*.mhd"));

num_files = size(all_mhd_files,1);

x = zeros(num_files,1);
y = zeros(num_files,1);
z = zeros(num_files,1);

for idx=1:num_files
    
    currfile = all_mhd_files(idx,:);

    % Read all lines of mhd file
    all_metadata = readlines(strcat(dir_path, currfile.name));

    % Find index that ElementSpacing is in
    find_elspace = strfind(all_metadata, "ElementSpacing");
    elspace_ind = find(~cellfun(@isempty,find_elspace));

    elspace_str = all_metadata(elspace_ind,:);
    element_spacing = split(elspace_str, " ");

    element_spacing = str2double(element_spacing(3:end,:));
    
    x(idx,:) = element_spacing(1);
    y(idx,:) = element_spacing(2);
    z(idx,:) = element_spacing(3);

end

avg_x = mean(x);
avg_y = mean(y);
avg_z = mean(z);

min_x = min(x);
min_y = min(y);
min_z = min(z);

max_x = max(x);
max_y = max(y);
max_z = max(z);