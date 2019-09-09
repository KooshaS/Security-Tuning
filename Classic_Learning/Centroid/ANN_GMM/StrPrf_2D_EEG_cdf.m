close all, clear all, clc, format compact


%%% Data Prepration %%%
load 'Dataset1.mat'

signal = zeros(106, 19200);
for i= 1:106
    % Channel F_PZ Data
    x = Raw_Data(i,2,:);
    signal(i,:) = reshape (x, [1, 19200]);
end

%%% Feature Extraction %%%
feature = zeros(2, 2, 239);
for i= 1:2
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, 1, j) = mean(signal_freq(8:13));
        feature(i, 2, j) = mean(signal_freq(14:30));
    end   
end

% Number of training samples of each class
N_trn = 180;

%%% Training Data and Lables %%%
x = [reshape(feature(1, :, 1:N_trn), [2, N_trn]) reshape(feature(2, :, 1:N_trn), [2, N_trn])]; % inputs
y = [zeros(1,N_trn) ones(1,N_trn)]; % outputs

% Plot input samples with PLOTPV (Plot perceptron input/target vectors)
figure(1)
plotpv(x,y);


% Create Network
net = network( ...
1, ... % numInputs, number of inputs,
2, ... % numLayers, number of layers
[1; 0], ... % biasConnect, numLayers-by-1 Boolean vector,
[1; 0], ... % inputConnect, numLayers-by-numInputs Boolean matrix,
[0 0; 1 0], ... % layerConnect, numLayers-by-numLayers Boolean matrix
[0 1] ... % outputConnect, 1-by-numLayers Boolean vector
);

% Configuring the Network
net.layers{1}.size = 3;
net.layers{1}.transferFcn = 'logsig';
% view(net);

net = configure(net,x,y);

% % Initial Network Response Without Training
% initial_output = net(inputs);

% Network Training
net.trainFcn = 'trainlm';
net.performFcn = 'mse';
net = train(net,x,y);











% hiddenLayerSize = 2;
% net = patternnet(hiddenLayerSize);
% % net = perceptron;
% net.trainParam.showWindow = false;
% net.trainParam.showCommandLine = false; 
% net = train(net,x,y);

% view(net);

% figure(1)
% plotpc(net.IW{1},net.b{1});
% hold on

% syms x1 x2
% f(x1,x2) = (x1 * net.IW{1}(1,1) + x2 * net.IW{1}(1,2) + net.b{1}(1)) * net.LW{2}(1) + ...
%            (x1 * net.IW{1}(2,1) + x2 * net.IW{1}(2,2) + net.b{1}(2)) * net.LW{2}(2) + ...
%            net.b{2} == 0;
% ezplot(f)

%%% Testing Data %%%
N_tst = 59;
x = [reshape(feature(1, :, 181:239), [2, 59]) reshape(feature(2, :, 181:239), [2, 59])]; % inputs

FR = 0;
for i = 1:N_tst
    if net(x(:, i)) == 1
        FR = FR + 1;
    end
end

FA = 0;
for i = 1:N_tst
    if net(x(:, 10 + i)) == 0
        FA = FA + 1;
    end
end

HTER = 100*(FA/N_tst + FR/N_tst)/2

% generate a grid
span = -500:1:2500;
% span = -2:.01:6;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';
% simulate neural network on a grid
aa = net(pp);
% translate output into [-1,1]
aa = -1 + 2*(aa>0);
% plot classification regions
figure(1)
mesh(P1,P2,reshape(aa,length(span),length(span))-5);
colormap cool

view(2)







