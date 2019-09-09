%%%%%%%%%% Using Gaussian Mixture Distribution for MI data modeling %%%%%%%%%%
clc, clear all


%%% Data Prepration %%%
load 'Dataset1.mat'

signal = zeros(106, 19200);
for i= 1:106
    % Channel F_PZ Data
%     x = Physionet_EEG_MI(i,23,:);
    x = Raw_Data(i,2,:);
    signal(i,:) = reshape (x, [1, 19200]);
end

%%% Feature Extraction %%%
feature = zeros(2, 239);
for i= 1:2
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, j) = mean(signal_freq(8:13));
    end   
end

%%% Training Phase %%%%
[muhat_sub1_trn,sigmahat_sub1_trn] = normfit(feature(1, 1:180));
[muhat_sub2_trn,sigmahat_sub2_trn] = normfit(feature(2, 1:180));
[muhat_w,sigmahat_w] = normfit([feature(1, 1:150) feature(2, 1:180)]);

%%% Testing Phase %%%
GMModel_sub1_tst = fitgmdist(feature(1, 181:239)',2);
GMModel_sub2_tst = fitgmdist(feature(2, 181:239)',2);

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 5*sigmahat_sub1_trn/changing_seed;
    FAR(1, i) = 100*(cdf(GMModel_sub2_tst, muhat_sub1_trn + thr) - cdf(GMModel_sub2_tst, muhat_sub1_trn - thr));
    security_bits(1, i) = log2(10*sigmahat_w/(2*thr));
end

thr = 0;
changing_seed = 1000;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 5*sigmahat_sub1_trn/changing_seed;
    FRR(1, i) = 100*(1 - cdf(GMModel_sub1_tst, muhat_sub1_trn + thr) + cdf(GMModel_sub1_tst, muhat_sub1_trn - thr));
end

HTER = (FAR + FRR)/2;



figure
yyaxis left
% t1 = 1:1000;
% t2 = 1:20:1000;
% vq2 = interp1(t1,HTER,t2,'spline');
t3 = 0.005:0.005:5;
% plot(t2,vq2);
% plot(t3,vq2);
plot(t3,HTER);

% plot(HTER)
xlabel('Threshold (\tau)');
ylabel('HTER(%)');

yyaxis right
t4 = 0.005:0.005:5;
plot(t4,security_bits)
ylabel('Security Strength (bits)');










