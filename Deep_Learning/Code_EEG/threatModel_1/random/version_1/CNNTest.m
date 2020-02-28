%%% Random Hyper-Parameters Optimization for CNN on Digits Data %%%
clc, clear all
% warning off;

%% Preparing the Dataset
digitDatasetPath = '..\..\..\dataset\EEG_Sub_1';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

rng(1);
numTrainFiles = 400;
[imdsTrain, imdsRest] = splitEachLabel(imds, numTrainFiles, 'randomize');
numValidFiles = 100;
[imdsValid, imdsTest] = splitEachLabel(imdsRest, numValidFiles, 'randomize');

layers = [
    imageInputLayer([32 32 1])
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValid, ...
    'ValidationFrequency',30, ...
    'Verbose',false);

net = trainNetwork(imdsTrain, layers, options);

YPred = classify(net, imdsValid);
YValidation = imdsValid.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)


