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
feature = zeros(2, 119);
for i= 1:2
    for j = 1:119     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, j) = mean(signal_freq(8:13));
    end   
end

%%% Training Phase %%%%
[muhat_sub1_trn,sigmahat_sub1_trn] = normfit(feature(1, 1:90));
[muhat_sub2_trn,sigmahat_sub2_trn] = normfit(feature(2, 1:90));
[muhat_w,sigmahat_w] = normfit([feature(1, 1:90) feature(2, 1:90)]);

%%% Testing Phase %%%
[muhat_sub1_tst,sigmahat_sub1_tst] = normfit(feature(1, 91:119));
[muhat_sub2_tst,sigmahat_sub2_tst] = normfit(feature(2, 91:119));

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 3*sigmahat_sub1_trn/changing_seed;
    FAR(1, i) = 100*(normcdf(muhat_sub1_trn + thr, muhat_sub2_tst, sigmahat_sub2_tst) - normcdf(muhat_sub1_trn - thr, muhat_sub2_tst, sigmahat_sub2_tst));
    security_bits(1, i) = log2(10*sigmahat_w/(2*thr));
end

thr = 0;
changing_seed = 1000;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 3*sigmahat_sub1_trn/changing_seed;
    FRR(1, i) = 100*(1 - normcdf(muhat_sub1_trn + thr, muhat_sub1_tst, sigmahat_sub1_tst) + normcdf(muhat_sub1_trn - thr, muhat_sub1_tst, sigmahat_sub1_tst));
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












