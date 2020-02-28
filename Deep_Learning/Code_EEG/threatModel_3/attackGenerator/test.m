clc, clear all

load EEG_dataset_Sub1

% for i = 1:1000
img = squeeze(XData(:, :, :, 1));
img2 = imnoise(img, 'gaussian', 0, 1);    
img3 = awgn(img, 20, 'measured');
img4 = round(img3);
    
% end



% I = double(imread('eight.tif'));
% imshow(I)
% 
% 
% J = imnoise(I,'salt & pepper',0.02);
% imshow(J)
% 
% K = imnoise(I,'gaussian', 0, 1);
% imshow(K)

