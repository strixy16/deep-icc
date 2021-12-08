% Script to rearrange MCRC directory 

main_dir = "../../Data/MCRC/";
orig_dir = strcat(main_dir, "originals/");

newDirs = {strcat(main_dir, "Liver/"), ...
           strcat(main_dir, "Tumor/"), ...
           strcat(main_dir, "Venous/"),...
           strcat(main_dir, "volume/")};

for idx=1:(size(newDirs,2))
    if ~exist(newDirs{idx}, 'dir')
        mkdir(newDirs{idx})
    end
end

all_dir_info = dir(strcat(orig_dir, "*preop*"));

tbl_all_dir_info = struct2table(all_dir_info);

cell_all_folder_names = natsort(tbl_all_dir_info{:,'name'});

for folder_idx = 1:size(cell_all_folder_names,1)
    img_dir = strcat(orig_dir, cell_all_folder_names{folder_idx}, "/");
    img_files = dir(strcat(img_dir, "*preop*"));
    
    for file_idx = 1:size(img_files,1)
        fname = img_files(file_idx).name;
        source = strcat(img_dir, fname);
        
        if contains(fname, "Liver")
            destination = newDirs{1};
            copyfile(source, destination);
        elseif contains(fname, "Tumor")
            destination = newDirs{2};
            copyfile(source, destination);
        elseif contains(fname, "Venous")
            destination = newDirs{3};
            copyfile(source, destination);
        elseif contains(fname, "volume")
            destination = newDirs{4};
            copyfile(source, destination);
        end
    end
end


