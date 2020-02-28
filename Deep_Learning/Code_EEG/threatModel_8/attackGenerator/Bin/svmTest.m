%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
clc, clear all
% warning off;

%% Preparing the Dataset
load '..\EEG_dataset_Sub1';

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

X = zeros(size(dataTestingRaw, 4), 512);
vectorizedImg = zeros(1, 512);
Y = cellstr(categorical(dataTestingLabel));

for i = 1:size(dataTestingRaw, 4)
    img = squeeze(dataTestingRaw(:, :, :, i));
    
    for l = 1:8
        for m = 1:8
            for n = 1:8
                vectorizedImg(1, (l - 1) * 64 + (m - 1) * 8 + n) = img(l, m, n);                                                
            end
        end
    end
    X(i, :) = vectorizedImg;
end

% rng(1); % For reproducibility
SVMModel = fitcsvm(X, Y, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, ...
    'KernelScale', 'auto');

CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);

%% Optimization (Hill-Climbing)
for i = 1:1000
    
    rndidx = randi([1 150]);
    img = imread(imdsTest.Files{rndidx}); 
    x0 = zeros(1, 784);
    for j = 1:28
        x0(1, (j - 1) * 28 + 1: 28 * j)= img(j, :);         
    end
    
    score_prev = matchscore(SVMModel, x0);
    itr = 10;
    while itr > 0        
        perturbation = randi([-1 1], 1, 784);
        x = x0 + perturbation;
        x = x + (x < 0);
        x = x - (x > 255);
        
        if score_prev < matchscore(SVMModel, x)
            score_prev = matchscore(SVMModel, x);
            x0 = x;
            itr = itr - 1;            
        end
    end
    
    perturbedImg = uint8(zeros(28, 28));
    for j = 1:28
        for k = 1:28
        perturbedImg(j, k) = x0(1, (j - 1) * 28 + k);
        end
    end
    
    imgName = sprintf('%03d', i - 1);
    imgDirectory = '..\..\DigitDataset_attack_v3\0\';
    imgPath = strcat(imgDirectory, imgName, '.png');
    imwrite(perturbedImg, imgPath)
    
end

