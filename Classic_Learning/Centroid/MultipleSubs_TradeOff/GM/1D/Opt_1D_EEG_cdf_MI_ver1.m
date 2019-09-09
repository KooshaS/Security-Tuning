clc, clear all
%%%%%%%%% MI Dataset %%%%%%%%%

%%% Data Prepration %%%
load 'Dataset1.mat'

sub = 106;
signal = zeros(106, 19200);
for i= 1:106
    % Channel F_PZ Data
%     x = Physionet_EEG_MI(i,23,:);
    x = Raw_Data(i,2,:);
    signal(i,:) = reshape (x, [1, 19200]);
end

%%% Feature Extraction %%%
feature = zeros(sub, 239);
for i= 1:sub
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, j) = mean(signal_freq(8:13));
    end   
end

%%% Training Phase %%%%
feature_trn = feature(:, 1:180);
[muhat_sub1_trn,sigmahat_sub1_trn] = normfit(feature(1, 1:180));
[muhat_sub2_trn,sigmahat_sub2_trn] = normfit(reshape(permute(feature_trn(2:sub, :), [2 1]), [1, 180*(sub - 1)]));

[muhat_w,sigmahat_w] = normfit(reshape(permute(feature_trn(1:sub, :), [2 1]), [1, 180*sub]));

%%% Testing Phase %%%
feature_tst = feature(:, 181:239);
[muhat_sub1_tst,sigmahat_sub1_tst] = normfit(feature(1, 181:239));
[muhat_sub2_tst,sigmahat_sub2_tst] = normfit(reshape(permute(feature_tst(2:sub, :), [2 1]), [1, 59*(sub - 1)]));

%%% CDFs Based on Security Bits %%%
K_upbound = 15;
FAR = zeros(1, K_upbound*100);

for K = 0.01:0.01:K_upbound
    threshold = 10*sigmahat_w/(2^(K + 1));
    FAR(1, int16(K*100)) = 100*(normcdf(muhat_sub1_trn + threshold, muhat_sub2_tst, sigmahat_sub2_tst) - normcdf(muhat_sub1_trn - threshold, muhat_sub2_tst, sigmahat_sub2_tst));
end

FRR = zeros(1, K_upbound*100);
for K = 0.01:0.01:K_upbound
    threshold = 10*sigmahat_w/(2^(K + 1));
    FRR(1, int16(K*100)) = 100*(1 - normcdf(muhat_sub1_trn + threshold, muhat_sub1_tst, sigmahat_sub1_tst) + normcdf(muhat_sub1_trn - threshold, muhat_sub1_tst, sigmahat_sub1_tst));
end

HTER = (FAR + FRR)/2;

K = 0.01:0.01:K_upbound;
plot(K, HTER);






