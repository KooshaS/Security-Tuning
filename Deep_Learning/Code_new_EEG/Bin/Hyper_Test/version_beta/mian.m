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
load '..\..\..\threatModel_2\EEGDataset_attack\EEG_dataset_attack';

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
lb = [1, 1, 1, 0, 0, 0, 1, 0, 1, 5];
ub = [4, 4, 4, 7, 7, 7, 5, 4, 7, 15];

fun = @(x)targetCNN(dataTrainingRaw, categorical(dataTrainingLabel), ...
    dataValidationRaw, categorical(dataValidationLabel), ...
    dataTestingRaw, categorical(dataTestingLabel), ...
    XDataAttack, [fix(x(1)), fix(x(2)), fix(x(3)), ...
    fix(x(4)), fix(x(5)), fix(x(6)), fix(x(7)), ...
    fix(x(8)), fix(x(9)), fix(x(10))]);
nvars = 10;

%%% Initial point
options = optimoptions('particleswarm', 'MaxIterations', 1000, 'MaxStallIterations', 300);

x0 = [2, 2, 2, 3, 3, 3, 2, 2, 3, 10];
options.InitialSwarmMatrix = x0; 

tic
[x, fval, exitflag] = particleswarm(fun, nvars, lb, ub, options);
exeTime = toc


