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
SVMModel = fitcsvm(X, Y, 'Standardize', true, 'KernelFunction', 'RBF', ...
    'KernelScale', 'auto');

% SVMModel = fitcsvm(X, Y, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, ...
%     'KernelScale', 'auto');

% CVSVMModel = crossval(SVMModel);
% classLoss = kfoldLoss(CVSVMModel);

%% Optimization (Hill-Climbing)
XDataAttack = zeros(8, 8, 8, 1000);
for i = 1:1000

%     rndidx = randi([1 200]);
%     img = squeeze(dataTestingRaw(:, :, :, rndidx));
    
    img = randi([-1024 1024], 8, 8, 8);
    x0 = zeros(1, 512);
    for l = 1:8
        for m = 1:8
            for n = 1:8
                x0(1, (l - 1) * 64 + (m - 1) * 8 + n) = img(l, m, n);                                              
            end
        end
    end
    
    fun = @(x)matchscore(SVMModel, fix(x));
    nvars = 512;

    lb = - 1024 * ones(1, 512);
    ub = 1024 * ones(1, 512);

    options = optimoptions('particleswarm', 'MaxIterations', 1000, 'MaxStallIterations', 100);
    options.InitialSwarmMatrix = x0;

    tic
    [x, fval, exitflag] = particleswarm(fun, nvars, lb, ub, options);
    exeTime = toc
    
    perturbedImg = zeros(8, 8, 8);
    for r = 1:8
        for s = 1:8
            for t = 1:8
                perturbedImg(r, s, t) = x(1, (r - 1) * 64 + (s - 1) * 8 + t);
            end
        end
    end
    
    XDataAttack(:, :, :, i) = fix(perturbedImg);
    
end

save('..\EEGdataset_attack\EEG_dataset_attack', 'XDataAttack');


