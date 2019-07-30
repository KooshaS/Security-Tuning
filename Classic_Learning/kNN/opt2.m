%%% One channel (O1) data for one subject %%%
clc, clear all

neighbours = 165;
accuracy = zeros(1, neighbours);
attSuccess = zeros(1, neighbours);

%% Data Prepration
load 'EEG_OpenEyes.mat'
load 'EEG_ClosedEyes.mat'
%%% O1 = 61 or O2 = 63
subject = 1;
XData = [reshape(OpenEyes(subject, 61, :), [1, 9600]) reshape(ClosedEyes(subject, 61, :), [1, 9600])]; 

xData = zeros(80, 120 * 2);
for i = 1:120 * 2
    xData(:, i) = XData(1, (i - 1) * 80 + 1:i * 80)';    
end

yData = [zeros(1, 120 ) ones(1, 120)];

%% Feature Extraction
% xData = preProcessing(xData);
xData = fourierTransform (xData);
% xData = waveletTransform (xData);

%% Separating Training and Testing Data
xData_trn = [xData(:, 1:80) xData(:, 121:200)]';
yData_trn = [yData(:, 1:80) yData(:, 121:200)]';
xData_tst = [xData(:, 81:120) xData(:, 201:240)]';
yData_tst = [yData(:, 81:120) yData(:, 201:240)]';  

for N = 1:neighbours
    
    N
    [accuracy(1, N), mdl] = kNNFunc(xData_trn, yData_trn, xData_tst, yData_tst, N);
    attSuccess(1, N) = kNNAtt(mdl);
        
end

accuracy(find(accuracy == 0)) = [];
attSuccess(find(attSuccess == 0)) = [];

plot(accuracy(1:160))
yyaxis right
plot(attSuccess(1:160))

