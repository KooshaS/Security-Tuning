clc, clear all


for i = 1:1000
    
    i
    r = uint8(randi([0 255], 32, 32));
    imgName = sprintf('%03d', i - 1);
    imgDirectory = '..\EEGDataset_attack\0\';
    imgPath = strcat(imgDirectory, imgName, '.png');
    imwrite(r, imgPath)
    
end


