close all, clear all, clc, format compact



% Initialization
world_size = 2;
targets = zeros(60*world_size,1);

% Making Class Labels
for i = 1:60
   targets(i) = -1; 
end

for i = 61:60*world_size
   targets(i) = 1; 
end

% Data prepration
load 'EEG_Rest_Open.mat'

% Training Data prepration
Signal = zeros(106, 9600);
for i= 1:106
    % Rest Session Data
    x = Physionet_EEG_MI(i,9,:);
    Signal(i,:) = reshape (x, [1, 9600]);
end

for i = 1:world_size
    for j = 1:60
        trn_signal = Signal(i,(j-1)*160+1:j*160);        
        S_F = abs(fft(trn_signal));
        S_F = S_F(2:160/2+1);
        trn_data((i-1)*60+j,:) = S_F(8:13); 
    end
end

% define inputs
P = trn_data(:,1:2)';
% P(1,:) = P(1,:) - mean(P(1,:));
% P(2,:) = P(2,:) - mean(P(2,:));
% P(1,:) = P(1,:) / std(P(1,:));
% P(2,:) = P(2,:) / std(P(2,:));

% define targets
T = targets';

% plot data
plot(P(1,1:60),P(2,1:60),'k+',P(1,61:120),P(2,61:120),'b*')
grid on
hold on

% create a neural network
net.trainParam.showWindow = false;
net = feedforwardnet(3);
% train net
net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio = 0; % validation set [%]
net.divideParam.testRatio = 0; % test set [%]

% train a neural network
[net,tr,Y,E] = train(net,P,T);
% show network
% view(net)

% test the NN
outputs = net(P);

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

% security performance
TA = sum(outputs(1:60) < 0);
FA = sum(outputs(61:120) < 0);
TR = sum(outputs(61:120) > 0);
FR = sum(outputs(1:60) > 0);

accuracy = (TA + TR)/(TA + TR + FA + FR);

efforts = (sum(aa==1)/size(aa,2))^(-1);









