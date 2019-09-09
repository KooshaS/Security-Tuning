close all, clear all, clc, format compact


load fisheriris.mat

% Number of training and testing samples of each class
N = 10;

%%% Training Data and Lables %%%
x = [meas(1:10, 1:2)' meas(51:60, 1:2)']; % inputs
y = [zeros(1,N) ones(1,N)]; % outputs

% Plot input samples with PLOTPV (Plot perceptron input/target vectors)
figure(1)
plotpv(x,y);

net = perceptron;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false; 
net = train(net,x,y);

% view(net);

figure(1)
plotpc(net.IW{1},net.b{1});
hold on

syms x1 x2
f(x1,x2) = x1 * net.IW{1}(1) + x2 * net.IW{1}(2) + net.b{1} == 0;
ezplot(f)

%%% Testing Data %%%
x = [meas(11:20, 1:2)' meas(61:70, 1:2)']; % inputs

FR = 0;
for i = 1:N
    if net(x(:, i)) == 1
        FR = FR + 1;
    end
end

FA = 0;
for i = 1:N
    if net(x(:, 10 + i)) == 0
        FA = FA + 1;
    end
end

HTER = (FA/10 + FR/10)/2
















