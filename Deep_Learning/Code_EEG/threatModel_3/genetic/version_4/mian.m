%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
%%% Adding learning rate as hyper-parameter %%%
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
lb = [1, 1, 1, 1, 1, 1, 1];
ub = [8, 8, 8, 100, 100, 100, 5];

fun = @(x)targetCNN(dataTrainingRaw, categorical(dataTrainingLabel), ...
    dataValidationRaw, categorical(dataValidationLabel), ...
    dataTestingRaw, categorical(dataTestingLabel), ...
    XDataAttack, x);

options = optimoptions('ga', 'MaxGenerations', 1000, ...
    'MaxStallGenerations', 300);

%% Initial Point
x0 = [4, 4, 4, 50, 50, 50, 2];
options.InitialPopulationMatrix = x0;

IntCon = [1, 2, 3, 4, 5, 6, 7];
nvars = 7;
tic
[x, fval, exitflag] = ga(fun, nvars, [], [], [], [], ...
    lb, ub, [], IntCon, options);
exeTime = toc


