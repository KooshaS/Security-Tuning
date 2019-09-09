clc, clear all
%%%%%%%%% MI Dataset %%%%%%%%%

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
[muhat_sub1_trn,sigmahat_sub1_trn] = normfit(feature(1, 1:150));
[muhat_sub2_trn,sigmahat_sub2_trn] = normfit(feature(2, 1:150));
[muhat_w,sigmahat_w] = normfit([feature(1, 1:90) feature(2, 1:150)]);

%%% Testing Phase %%%
GMModel_sub1_tst = fitgmdist(feature(1, 181:239)',2);
GMModel_sub2_tst = fitgmdist(feature(2, 181:239)',2);

%%% CDFs Based on Security Bits %%%
K_upbound = 15;
FAR = zeros(1, K_upbound*100);

for K = 0.01:0.01:K_upbound
    threshold = 10*sigmahat_w/(2^(K + 1));    
    FAR(1, int16(K*100)) = 100*(cdf(GMModel_sub2_tst, muhat_sub1_trn + threshold) - cdf(GMModel_sub2_tst, muhat_sub1_trn - threshold));
end

FRR = zeros(1, K_upbound*100);
for K = 0.01:0.01:K_upbound
    threshold = 10*sigmahat_w/(2^(K + 1));
    FRR(1, int16(K*100)) = 100*(1 - cdf(GMModel_sub1_tst, muhat_sub1_trn + threshold) + cdf(GMModel_sub1_tst, muhat_sub1_trn - threshold));    
end

HTER = (FAR + FRR)/2;

K = 0.01:0.01:K_upbound;
plot(K, HTER);








