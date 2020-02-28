%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
clc, clear all
% warning off;

%% Preparing the Dataset
load 'EEG_dataset_Sub1';

%%% Spiliting data as training and testing %%%
p = 2 / 3;
N = size(YData, 1);     % total number of rows
tf = false(N, 1);
tf(1:round(p * N)) = true;
rng(1);                 % For same test data in all experiments
tf = tf(randperm(N));   % randomise order

dataTrainingRaw = XData(:, :, :, tf);
dataTrainingLabel = YData(tf, :);
dataRemainingRaw = XData(:, :, :,~tf);
dataRemainingLabel = YData(~tf, :);

% Seperating test from validation data %
p = 1 / 2;
N = size(dataRemainingLabel, 1);    % total number of rows
tf = false(N, 1);
tf(1:round(p * N)) = true;
rng(2);                             % For same test data in all experiments
tf = tf(randperm(N));               % randomise order

dataValidationRaw = dataRemainingRaw(:, :, :, tf);
dataValidationLabel = dataRemainingLabel(tf, :);
dataTestingRaw = dataRemainingRaw(:, :, :,~tf);
dataTestingLabel = dataRemainingLabel(~tf, :);

X = zeros(size(dataTestingRaw, 4), 200);
vectorizedImg = zeros(1, 200);
Y = cellstr(categorical(dataTestingLabel));

for i = 1:size(dataTestingRaw, 4)
    img = squeeze(dataTestingRaw(:, :, :, i));
    
    imgResized = zeros(5, 5, 8);
    for v = 1:8
       imgResized(:, :, v) = imresize(img(:, :, v), [5 5]); 
    end

    for l = 1:8
        for m = 1:5
            for n = 1:5
                vectorizedImg(1, (l - 1) * 25 + (m - 1) * 5 + n) = imgResized(m, n, l);                                                
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
    
    perturbedImg = zeros(5, 5, 8);
    for r = 1:8
        for s = 1:5
            for t = 1:5
                perturbedImg(s, t, r) = x(1, (r - 1) * 25 + (s - 1) * 5 + t);
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



