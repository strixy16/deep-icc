function intMask = generateMask(I)
    % Convert image to double
%     I = double(I);
    % Generate mask excluding values outside of this range (Hounsfield
    % units)
    % This range includes only soft tissue
    Imask = I<-100|I>300;
    % Flips black and white in the mask
    ImaskInv = imcomplement(Imask);
    clear Imask I
    % Apply mask to image to separate tumor from background
%     Inew = I .* ImaskInv;
    intMask = int8(ImaskInv);
    % Make background NaN instead of 0s
%     Inew(Inew==0) = NaN;
%     out = Inew;
end