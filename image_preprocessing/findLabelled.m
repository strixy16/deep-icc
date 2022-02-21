function findLabelled(conf_f)
    
    if ischar(conf_f) || isstring(conf_f)
        conf_f = str2func(conf_f);
        options = conf_f();
    elseif isstruct(conf_f)
        options = conf_f();
    else
        error("Input must be struct or name of .m config file")
    end

    all_file_info = dir(strcat(options.all_file_dir, options.search_string));

    tbl_all_file_info = struct2table(all_file_info);

    cell_all_file_names = natsort(tbl_all_file_info{:,'name'});

    img_labels = readtable(options.Labels);

%     files_to_move = cell(0, size(img_labels,1)*2);
    
    % Making sure labelled_only directory exists 
    if ~exist(options.destination_dir, 'dir')
        mkdir(options.destination_dir)
    end

    for label_idx = 1:size(img_labels, 1)
        patient_ID = img_labels.ScoutID(label_idx);

        patient_ID = strcat(patient_ID, '_');

        labelled_patient_idx = contains(cell_all_file_names, patient_ID);

        files_to_move = cell_all_file_names(labelled_patient_idx);
        
        % Display patient ID if this patient has a label but no images
        if size(files_to_move,1) == 0
            disp(patient_ID);
        end

%         for file_idx = 1:size(files_to_move, 1)
%             source = strcat(options.all_file_dir, files_to_move{file_idx});
%             destination = strcat(options.destination_dir, files_to_move{file_idx});
% 
%             status = copyfile(source, destination);
%         end

    end
end