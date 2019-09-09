% clc, clear all
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

%%% Testing Phase %%%
feature_tst = feature(:, 181:239);
[muhat_sub1_tst,sigmahat_sub1_tst] = normfit(feature(1, 181:239));
[muhat_sub2_tst,sigmahat_sub2_tst] = normfit(reshape(permute(feature_tst(2:sub, :), [2 1]), [1, 59*(sub - 1)]));

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 10*sigmahat_sub1_trn/changing_seed;
    FAR(1, i) = 100*(normcdf(muhat_sub1_trn + thr, muhat_sub2_tst, sigmahat_sub2_tst) - normcdf(muhat_sub1_trn - thr, muhat_sub2_tst, sigmahat_sub2_tst));
    
    security_bits(1, i) = log2(20*sigmahat_w/(2*thr));
end

thr = 0;
changing_seed = 1000;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 10*sigmahat_sub1_trn/changing_seed;
    FRR(1, i) = 100*(1 - normcdf(muhat_sub1_trn + thr, muhat_sub1_tst, sigmahat_sub1_tst) + normcdf(muhat_sub1_trn - thr, muhat_sub1_tst, sigmahat_sub1_tst));
end

HTER = (FAR + FRR)/2;

CCR = (100 - FAR + 100 - FRR)/2;











