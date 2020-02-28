%%% 10 subjects authentication in a 106 people world %%%
clc, clear all


load 'EEG_dataset_Sub1.mat'

%%% Spiliting data as training and testing %%%
p = 2 / 3;
N = size(YData, 1);    % total number of rows
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


% Defining the CNN %
layers = [
    imageInputLayer([8 8 8])
    
    convolution2dLayer(3, 32, 'Padding', 1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
%     convolution2dLayer(3, 32, 'Padding', 1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2, 'Stride', 2)
%     
%     convolution2dLayer(3, 32, 'Padding', 1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2, 'Stride', 2)
%     
%     convolution2dLayer(3, 32, 'Padding', 1)
%     batchNormalizationLayer
%     reluLayer
   
    fullyConnectedLayer(512)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% options = trainingOptions('sgdm', ...
%     'MaxEpochs',8, ...
%     'ValidationData',{dataValidationRaw, categorical(dataValidationLabel)}, ...
%     'ValidationFrequency',30, ...      
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{dataValidationRaw, categorical(dataValidationLabel)}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(dataTrainingRaw, categorical(dataTrainingLabel), layers, options);

YPred = classify(net, dataTestingRaw);

accuracy = sum(YPred == categorical(dataTestingLabel))/numel(dataTestingLabel)





