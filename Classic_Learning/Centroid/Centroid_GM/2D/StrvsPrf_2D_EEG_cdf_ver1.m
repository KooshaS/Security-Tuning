%%%%%%%%%% Square Integration Domain %%%%%%%%%%
clc, clear all


%%% Data Prepration %%%
load 'EEG_Rest_Open.mat'

signal = zeros(106, 9600);
for i= 1:106
    % Channel F_PZ Data
    x = Physionet_EEG_MI(i,23,:);
    signal(i,:) = reshape (x, [1, 9600]);
end

%%% Feature Extraction %%%
feature = zeros(2, 2, 119);
for i= 1:2
    for j = 1:119     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, 1, j) = mean(signal_freq(8:13));
        feature(i, 2, j) = mean(signal_freq(14:30));
    end   
end

%%% Training Phase %%%%
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(feature(1, 1, 1:90), [1, 90]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(feature(1, 2, 1:90), [1, 90]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(feature(2, 1, 1:90), [1, 90]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(feature(2, 2, 1:90), [1, 90]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit([reshape(feature(1, 1, 1:90), [1, 90]) reshape(feature(2, 1, 1:90), [1, 90])]);
[muhat_w_2,sigmahat_w_2] = normfit([reshape(feature(1, 2, 1:90), [1, 90]) reshape(feature(2, 2, 1:90), [1, 90])]);

%%% Testing Phase %%%
[muhat_sub1_tst1,sigmahat_sub1_tst1] = normfit(reshape(feature(1, 1, 91:119), [1, 29]));
[muhat_sub1_tst2,sigmahat_sub1_tst2] = normfit(reshape(feature(1, 2, 91:119), [1, 29]));

[muhat_sub2_tst1,sigmahat_sub2_tst1] = normfit(reshape(feature(2, 1, 91:119), [1, 29]));
[muhat_sub2_tst2,sigmahat_sub2_tst2] = normfit(reshape(feature(2, 2, 91:119), [1, 29]));

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 3*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed; 
    mu = [muhat_sub2_tst1 muhat_sub2_tst2];
    Sigma = cov(reshape(feature(2, 1, 91:119), [29, 1]), reshape(feature(2, 2, 91:119), [29, 1]));
    [F,err] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu,Sigma);
    FAR(1, i) = 100*F;
%     security_bits(1, i) = log2(pi*(3*max(sigmahat_w_1, sigmahat_w_2))^2/(pi*thr^2));
security_bits(1, i) = log2(10*sigmahat_w_1*10*sigmahat_w_2/(pi*thr^2));
end

thr = 0;
changing_seed = 1000;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 3*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed; 
    mu = [muhat_sub1_tst1 muhat_sub1_tst2];
    Sigma = cov(reshape(feature(1, 1, 91:119), [29, 1]), reshape(feature(1, 2, 91:119), [29, 1]));
    [F,err] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu,Sigma);
    FRR(1, i) = 100*(1 - F);
end

HTER = (FAR + FRR)/2;

figure
yyaxis left
% t1 = 1:1000;
% t2 = 1:20:1000;
% vq2 = interp1(t1,HTER,t2,'spline');
t3 = 0.003:0.003:3;
% plot(t2,vq2);
% plot(t3,vq2);
plot(t3,HTER);

% plot(HTER)
xlabel('Threshold');
ylabel('HTER(%)');

yyaxis right
t4 = 0.003:0.003:3;
plot(t4,security_bits)
ylabel('Security Strength (bits)');







