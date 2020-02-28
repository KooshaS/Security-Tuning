%%% Arranging EEG signals in Frequency Domain for 100 Subjects and 64 channels %%%
clc, clear all

sub = 1;
load 'EEG_OpenEyes.mat'
load 'EEG_ClosedEyes.mat'

xData = zeros(64 * 6, 100 * 120);
yData = zeros(1, 100 * 120);

subject_EEG = squeeze(ClosedEyes(sub, :, :));
img = zeros(32, 32);
subImg = zeros(8, 8);

for itr = 1:600
    for i = 1:16
        for j = 1:8
            for k = 1:8
                subImg(j, k) = subject_EEG((j - 1) * 8 + k, (itr - 1) * 16 + i);
            end
        end

        rowIdx = idivide(int16(i - 1),4);
        colIdx = mod(i - 1, 4);
        img(rowIdx * 8 + 1:rowIdx * 8 + 8, colIdx * 8 + 1:colIdx * 8 + 8) = subImg;

    end
    
    %% Converting to Image Data
    img = uint8(rescale(img, 0, 255));
    
    %% Storing the Data
    imgName = sprintf('%03d', itr - 1);
    imgDirectory = 'EEG_Sub_1\1\';
    imgPath = strcat(imgDirectory, imgName, '.png');
    imwrite(img, imgPath)

end


