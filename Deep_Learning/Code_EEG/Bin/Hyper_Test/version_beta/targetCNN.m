function prf = targetCNN(XTrain, YTrain, XValid, YValid, XTest, YTest, XDataAttack, P)

    %% Storing the results in the workspace
    global optimizationResults
    global itrNumb

    P(4) = 2 ^ P(4);
    P(5) = 2 ^ P(5);
    P(6) = 2 ^ P(6);
    
    rng default
    %% Setting up the Network
    layers = [
        imageInputLayer([8 8 8])

        convolution2dLayer(P(1), P(4), 'Padding', 'same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(P(2), P(5), 'Padding', 'same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(P(3), P(6), 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];

    options = trainingOptions('sgdm', ...
        'InitialLearnRate', (0.1) ^ P(7), ...
        'MaxEpochs', 2 ^ P(8), ...
        'MiniBatchSize', 2 ^ P(9), ...
        'Shuffle', 'every-epoch', ...
        'ValidationData',{XValid, YValid}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false);

    %% Training the Network
    net = trainNetwork(XTrain, YTrain, layers, options);

    %% Accuracy Test
    [acc, history] = accuracyEval(net, XTest, YTest, P(10));
    
    %% Security Test
%     digitDatasetAttackPath = 'EEGDataset_attack';
%     imdsAtt = imageDatastore(digitDatasetAttackPath, ...
%         'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    sec = securityEval(net, history, XDataAttack, P(10));
    
    %% Results
    prf = - (0.5 * acc - 0.5 * sec);

    fprintf('%3d\t\t|%3d\t|%3d\t|%3d\t|%3d\t|%3d\t|%3d\t|%3d\t|%3d\t|%3d\t|%3d\t|%4.2f\t\t|%4.2f\t\t|%4.2f\t\t|\n',...
            itrNumb, P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(8), ...
            P(9), P(10), acc, sec, - prf)
    
    optimizationResults = [optimizationResults; {itrNumb, P(1), P(2), ...
        P(3), P(4), P(5), P(6), P(7), P(8), P(9), P(10), acc, sec, - prf}];
    
    itrNumb = itrNumb + 1;

end


