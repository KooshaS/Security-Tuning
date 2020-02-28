%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
clc, clear all
% warning off;

%% Preparing the Dataset
% digitDatasetPath = '..\dataset\EEG_Sub_1';
% imds = imageDatastore(digitDatasetPath, ...
%     'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% rng(1);
% numTrainFiles = 400;
% [imdsTrain, imdsRest] = splitEachLabel(imds, numTrainFiles, 'randomize');
% numValidFiles = 100;
% [imdsValid, imdsTest] = splitEachLabel(imdsRest, numValidFiles, 'randomize');

load '..\..\..\dataset\EEG_dataset_Sub1';
load '..\..\EEGDataset_attack\EEG_dataset_attack';

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

%% Optimization
global optimizationResults
global itrNumb;
itrNumb = 1;

%% Hyper-Parameter Optimization (using Random Search with Search Bounds)
tic

%% Initial Point
x0 = [4, 4, 4, 512, 512, 512];

%%% Search bounds
lb = [1, 1, 1, 1, 1, 1];
ub = [8, 8, 8, 1024, 1024, 1024];

fun = @(x)targetCNN(dataTrainingRaw, categorical(dataTrainingLabel), ...
    dataValidationRaw, categorical(dataValidationLabel), ...
    dataTestingRaw, categorical(dataTestingLabel), ...
    XDataAttack, [fix(x(1)), fix(x(2)), fix(x(3)), ...
    fix(x(4)), fix(x(5)), fix(x(6))]);

options = optimoptions('simulannealbnd', 'MaxIterations', 1000, 'MaxStallIterations', 500);

[x, fval, exitFlag, output] = simulannealbnd(fun, x0, lb, ub, options);

exeTime = toc


