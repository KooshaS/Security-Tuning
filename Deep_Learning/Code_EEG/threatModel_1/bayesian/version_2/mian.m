%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
clc, clear all
% warning off;

%% Preparing the Dataset
% digitDatasetPath = '\\client\c$\Users\Koosha\Dropbox (ASU)\BrainResearch\ACTVE PAPERS\ITII2019_SecTuning\Code_new\case_study\DigitDataset_01';
% digitDatasetPath = '..\..\..\dataset\EEG_Sub_1';
% imds = imageDatastore(digitDatasetPath, ...
%     'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
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

%% Hyper-Parameter Optimization (using Bayesian optimization)
num1 = optimizableVariable('n1',[1,4],'Type','integer');
num2 = optimizableVariable('n2',[1,4],'Type','integer');
num3 = optimizableVariable('n3',[1,4],'Type','integer');
num4 = optimizableVariable('n4',[1,4],'Type','integer');
num5 = optimizableVariable('n5',[1,4],'Type','integer');
num6 = optimizableVariable('n6',[1,128],'Type','integer');
num7 = optimizableVariable('n7',[1,128],'Type','integer');
num8 = optimizableVariable('n8',[1,128],'Type','integer');

fun = @(x)targetCNN(dataTrainingRaw, categorical(dataTrainingLabel), ...
    dataValidationRaw, categorical(dataValidationLabel), ...
    dataTestingRaw, categorical(dataTestingLabel), ...
    XDataAttack, [x.n1, x.n2, x.n3, x.n4, x.n5, x.n6, x.n7, x.n8]);

tic
results = bayesopt(fun, [num1, num2, num3, num4, num5, num6, num7, num8], 'Verbose', 0, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObj', 1000, ...
    'IsObjectiveDeterministic', true, ...
    'UseParallel', false);
exeTime = toc;

save 'results'


