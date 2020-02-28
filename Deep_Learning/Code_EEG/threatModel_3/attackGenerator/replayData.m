%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
clc, clear all
% warning off;

load EEG_dataset_Sub1
XDataAttack = zeros(8, 8, 8, 1000);

for j = 1:10
    for i = 1:100
        img = squeeze(XData(:, :, :, i));
        img = awgn(img, j, 'measured');
        XDataAttack(:, :, :, (j - 1) * 100 + i) = img;
    end
end

