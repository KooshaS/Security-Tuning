%%%%%%%%%% Square Integration Domain Using MI %%%%%%%%%%
clc, clear all


%%% Data Prepration %%%
load 'Dataset1.mat'

signal = zeros(106, 19200);
for i= 1:106
    % Channel F_PZ Data
    x = Raw_Data(i,2,:);
    signal(i,:) = reshape (x, [1, 19200]);
end

%%% Feature Extraction %%%
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

%%% Training Phase %%%%
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(feature(1, 1, 1:150), [1, 150]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(feature(1, 2, 1:150), [1, 150]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(feature(2, 1, 1:150), [1, 150]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(feature(2, 2, 1:150), [1, 150]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit([reshape(feature(1, 1, 1:150), [1, 150]) reshape(feature(2, 1, 1:150), [1, 150])]);
[muhat_w_2,sigmahat_w_2] = normfit([reshape(feature(1, 2, 1:150), [1, 150]) reshape(feature(2, 2, 1:150), [1, 150])]);

%%% Testing Phase %%%
[muhat_sub1_tst1,sigmahat_sub1_tst1] = normfit(reshape(feature(1, 1, 151:239), [1, 89]));
[muhat_sub1_tst2,sigmahat_sub1_tst2] = normfit(reshape(feature(1, 2, 151:239), [1, 89]));

[muhat_sub2_tst1,sigmahat_sub2_tst1] = normfit(reshape(feature(2, 1, 151:239), [1, 89]));
[muhat_sub2_tst2,sigmahat_sub2_tst2] = normfit(reshape(feature(2, 2, 151:239), [1, 89]));

%%% CDFs Based on Security Bits %%%
K_upbound = 25;
FAR = zeros(1, K_upbound*100);
for K = 0.01:0.01:K_upbound
    threshold = sqrt((10*sigmahat_w_1*10*sigmahat_w_2)/(pi*2^K));
    mu = [muhat_sub2_tst1 muhat_sub2_tst2];
    Sigma = cov(reshape(feature(2, 1, 151:239), [89, 1]), reshape(feature(2, 2, 151:239), [89, 1]));
    [F,err] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu,Sigma);
    FAR(1, int16(K*100)) = 100*F;   
end

FRR = zeros(1, K_upbound*100);
for K = 0.01:0.01:K_upbound
    threshold = sqrt((10*sigmahat_w_1*10*sigmahat_w_2)/(pi*2^K));
    mu = [muhat_sub1_tst1 muhat_sub1_tst2];
    Sigma = cov(reshape(feature(1, 1, 151:239), [89, 1]), reshape(feature(1, 2, 151:239), [89, 1]));
    [F,err] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu,Sigma);
    FRR(1, int16(K*100)) = 100*(1 - F);    
end

HTER = (FAR + FRR)/2;

K = 0.01:0.01:K_upbound;
plot(K, HTER);






