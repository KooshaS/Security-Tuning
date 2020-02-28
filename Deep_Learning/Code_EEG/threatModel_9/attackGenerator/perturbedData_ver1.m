%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
clc, clear all
% warning off;

%% Preparing the Dataset
load 'EEG_dataset_Sub2';

X = zeros(size(XData, 4), 288);
vectorizedImg = zeros(1, 288);
Y = cellstr(categorical(YData));

for i = 1:size(XData, 4)
    img = squeeze(XData(:, :, :, i));
    
    imgResized = zeros(6, 6, 8);
    for v = 1:8
       imgResized(:, :, v) = imresize(img(:, :, v), [6 6]); 
    end
    
    for l = 1:8
        for m = 1:6
            for n = 1:6
                vectorizedImg(1, (l - 1) * 36 + (m - 1) * 6 + n) = imgResized(m, n, l);                                                
            end
        end
    end
    X(i, :) = vectorizedImg;
end

[sigma, mu, ~, ~] = robustcov(X);

%% Generative Model
XDataAttack = zeros(8, 8, 8, 1000);
for i = 1:1000
    i
    x = mvnrnd(mu, sigma, 1);
    
    perturbedImg = zeros(6, 6, 8);
    for r = 1:8
        for s = 1:6
            for t = 1:6
                perturbedImg(s, t, r) = x(1, (r - 1) * 36 + (s - 1) * 6 + t);
            end
        end
    end
    
    perturbedImgResized = zeros(8, 8, 8);
    for v = 1:8
       perturbedImgResized(:, :, v) = imresize(perturbedImg(:, :, v), [8 8]); 
    end
    
    XDataAttack(:, :, :, i) = fix(perturbedImgResized);
    
end

save('..\EEGdataset_attack\EEG_dataset_attack', 'XDataAttack');



