function new_preprocessMHA(conf_f)

    if ischar(conf_f) || isstring(conf_f)
        conf_f = str2func(conf_f);
        options = conf_f();   
    elseif isstruct(conf_f)
        options = conf_f();
    else
       error("Input must be struct or name of .m config file") 
    end
    
    % Getting list of MHD tumor files
    baseDirs = dir(strcat(options.ImageLoc, "*Tumor*.mhd"));
    
    % Counter for loop
    nData = size(baseDirs, 1);
    
    % Recording max height and width for crop/rescaling
    maxHeight = 0;
    maxWidth = 0;
    
    % Initiate container for processed slice images
    procImages = cell(nData, 1);
    
    % Reading each individual MHD file to find the max height and width
    % from the images
    for currFile = 1:nData
        fprintf('Computing Size for %i \n', currFile)
        filename = strcat(options.ImageLoc, baseDirs(currFile).name);
        info = mha_read_header(filename);
        vol = double(mha_read_volume(info));
        [procVol, maskVol] = ProcessImage(vol);
        
        % Sum the volume by the 3rd dimension to get largest tumour
        % dimensions
        max_tumor_mask = sum(maskVol, 3);
        
        % Sum the rows to get max width
        max_tumor_cols = sum(max_tumor_mask, 1);
        non_zero_cols = find(max_tumor_cols);
        temp_width = non_zero_cols(end) - non_zero_cols(1);
        
        % sum the rows to get max height
        max_tumor_rows = sum(max_tumor_mask, 2);
        non_zero_rows = find(max_tumor_rows);
        temp_height = non_zero_rows(end) - non_zero_cols(1);
        
        % Check if width and height are larger than existing
        if temp_width > maxWidth
            maxWidth = temp_width;
        end
        if temp_height > maxHeight
            maxHeight = temp_height;
        end
        
        % Sum processed volume by 1st and 2nd dimension
        % Result is 1 if tumor pixels in slice, 0 if not
        temp_tumor_marker = sum(maskVol,2);
        tumor_marker = sum(temp_tumor_marker,1);
        
        tumor_slice_ind = find(tumor_marker);
        
        tumor_slices = procVol(:,:,tumor_slice_ind);
        procImages{currFile} = tumor_slices;
        
    end
    
    % Second loop
    fprintf("Finished first loop");
    
    

end