function new_preprocessMHA(conf_f)

%     if ischar(conf_f) || isstring(conf_f)
%         conf_f = str2func(conf_f);
%         options = conf_f();   
%     elseif isstruct(conf_f)
%         options = conf_f();
%     else
%        error("Input must be struct or name of .m config file") 
%     end
    options = erasmus_tumors();
    
    % Getting list of MHD tumor files
    baseDirs = dir(strcat(options.ImageLoc, "*Tumor*.mhd"));
    
    % Counter for loop
    nData = size(baseDirs, 1);
    
    % Recording max height and width for crop/rescaling
    maxHeight = 0;
    maxWidth = 0;
    
    % Initiate container for processed slice images
    procImages = cell(nData, 1);
    tumorCenters = zeros(nData, 2);
    
    % Reading each individual MHD file to find the max height and width
    % from the images
    for currFile = 1:nData
        fprintf('Computing Size for %i \n', currFile)
        filename = strcat(options.ImageLoc, baseDirs(currFile).name);
        info = mha_read_header(filename);
        vol = double(mha_read_volume(info));
        [procVol, maskVol] = new_ProcessImage(vol);
        
        % Sum the volume by the 3rd dimension to get largest tumour
        % dimensions
        max_tumor_mask = sum(maskVol, 3);
        
        % Sum the rows to get max width
        max_tumor_cols = sum(max_tumor_mask, 1);
        % Find the non-zero elements in the row (edges of largest tumor)
        non_zero_cols = find(max_tumor_cols);
        % Find the distance between the left and right edges of the tumor
        temp_width = non_zero_cols(end) - non_zero_cols(1);
        
        % sum the rows to get max height
        max_tumor_rows = sum(max_tumor_mask, 2);
        % Find the non-zero elements in the column (edges of largest tumor)
        non_zero_rows = find(max_tumor_rows);
        % Find the distance between the upper and lower edges of the tumor
        temp_height = non_zero_rows(end) - non_zero_rows(1);
        
        % TODO: FIND THE CENTER OF THE TUMOR AND STORE FOR SECOND LOOP
        % Find the center x coordinate of the tumor
        ctr_x = non_zero_cols(1) + floor(temp_width/2);
        % Find the center y coordinate of the tumor
        ctr_y = non_zero_rows(1) + floor(temp_height/2);
        tumorCenters(currFile,:) = [ctr_x, ctr_y];
        
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
        tumor_marker = reshape(tumor_marker, [size(tumor_marker,3),1]);
        
        % Finding slice indices that have tumor pixels
        tumor_slice_ind = tumor_marker > 0;
%         tumor_slice_ind = find(tumor_marker);
        
        % Selecting out tumor slices
        tumor_slices = procVol(:,:,tumor_slice_ind);
        
        % Storing only tumor slices for cropping
        procImages{currFile} = tumor_slices;
        
    end
    
    % Second loop to crop images based on max height and width
    
    % TODO: have to decide if we want the cropping to end up with the tumor
    % piece centered in the image, in which case have to loop over every
    % slice individually and find the center
    % OR if it's ok to have the tumor not centered, then can just use the
    % center calculated in the first loop and crop the whole 3D volume at
    % once
    
    for currFile = 1:nData
        fprintf('Cropping images for %i \n', currFile)
        vol = procImages{currFile};
        
        % Need to find center of the tumor 
        % use the max mask again -
    end

end