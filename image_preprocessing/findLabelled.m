
all_file_dir = "../../Data/cholangio/AllSources/Tumor/";
labels_file = "../../Data/RFS_Scout.xlsx";
destination_dir = "../../Data/cholangio/labelled_only/tumors/";

all_file_info = dir(strcat(all_file_dir, "*Tumor*"));

tbl_all_file_info = struct2table(all_file_info);

cell_all_file_names = natsort(tbl_all_file_info{:,'name'});

img_labels = readtable(labels_file);

files_to_move = cell(0, size(img_labels,1)*2);

for label_idx = 1:size(img_labels, 1)
    patient_ID = img_labels.ScoutID(label_idx);
    
    patient_ID = strcat(patient_ID, '_');
    
    labelled_patient_idx = contains(cell_all_file_names, patient_ID);
    
    files_to_move = cell_all_file_names(labelled_patient_idx);
    
    for file_idx = 1:size(files_to_move, 1)
        source = strcat(all_file_dir, files_to_move{file_idx});
        destination = strcat(destination_dir, files_to_move{file_idx});
        
        status = copyfile(source, destination);
    end
    
end