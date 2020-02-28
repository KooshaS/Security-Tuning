%%%%%%%%%% Square Integration Domain Using MI %%%%%%%%%%
% clc, clear all


%%% Data Prepration %%%
load 'Dataset1.mat'

sub = 106;
signal = zeros(106, 19200);
for i= 1:106
    % Channel F_PZ Data
    x = Raw_Data(i,2,:);
    signal(i,:) = reshape (x, [1, 19200]);
end

%%% Feature Extraction %%%
%%% Feature Extraction %%%
feature = zeros(sub, 5, 239);
for i= 1:sub
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, 1, j) = mean(signal_freq(1:3));
        feature(i, 2, j) = mean(signal_freq(4:7));
        feature(i, 3, j) = mean(signal_freq(8:13));
        feature(i, 4, j) = mean(signal_freq(14:30));
        feature(i, 5, j) = mean(signal_freq(31:40));        
    end   
end

%%% Training Phase %%%%
feature_trn = feature(:, :, 1:180);
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(permute(feature_trn(1, 1, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(permute(feature_trn(1, 2, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn3,sigmahat_sub1_trn3] = normfit(reshape(permute(feature_trn(1, 3, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn4,sigmahat_sub1_trn4] = normfit(reshape(permute(feature_trn(1, 4, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn5,sigmahat_sub1_trn5] = normfit(reshape(permute(feature_trn(1, 5, 1:180), [2 1 3]), [1, 180]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(permute(feature_trn(2:sub, 1, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(permute(feature_trn(2:sub, 2, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn3,sigmahat_sub2_trn3] = normfit(reshape(permute(feature_trn(2:sub, 3, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn4,sigmahat_sub2_trn4] = normfit(reshape(permute(feature_trn(2:sub, 4, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn5,sigmahat_sub2_trn5] = normfit(reshape(permute(feature_trn(2:sub, 5, 1:180), [3 2 1]), [1, 180*(sub - 1)]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit(reshape(permute(feature_trn(1:sub, 1, 1:180), [3 2 1]), [1, 180*sub]));
[muhat_w_2,sigmahat_w_2] = normfit(reshape(permute(feature_trn(1:sub, 2, 1:180), [3 2 1]), [1, 180*sub]));
[muhat_w_3,sigmahat_w_3] = normfit(reshape(permute(feature_trn(1:sub, 3, 1:180), [3 2 1]), [1, 180*sub]));
[muhat_w_4,sigmahat_w_4] = normfit(reshape(permute(feature_trn(1:sub, 4, 1:180), [3 2 1]), [1, 180*sub]));
[muhat_w_5,sigmahat_w_5] = normfit(reshape(permute(feature_trn(1:sub, 5, 1:180), [3 2 1]), [1, 180*sub]));

%%% Testing Phase %%%
feature_tst = feature(:, :, 181:239);
[muhat_sub1_tst1,sigmahat_sub1_tst1] = normfit(reshape(permute(feature_tst(1, 1, :), [2 1 3]), [1, 59]));
[muhat_sub1_tst2,sigmahat_sub1_tst2] = normfit(reshape(permute(feature_tst(1, 2, :), [2 1 3]), [1, 59]));
[muhat_sub1_tst3,sigmahat_sub1_tst3] = normfit(reshape(permute(feature_tst(1, 3, :), [2 1 3]), [1, 59]));
[muhat_sub1_tst4,sigmahat_sub1_tst4] = normfit(reshape(permute(feature_tst(1, 4, :), [2 1 3]), [1, 59]));
[muhat_sub1_tst5,sigmahat_sub1_tst5] = normfit(reshape(permute(feature_tst(1, 5, :), [2 1 3]), [1, 59]));

[muhat_sub2_tst1,sigmahat_sub2_tst1] = normfit(reshape(permute(feature_tst(2:sub, 1, :), [3 2 1]), [1, 59*(sub - 1)]));
[muhat_sub2_tst2,sigmahat_sub2_tst2] = normfit(reshape(permute(feature_tst(2:sub, 2, :), [3 2 1]), [1, 59*(sub - 1)]));
[muhat_sub2_tst3,sigmahat_sub2_tst3] = normfit(reshape(permute(feature_tst(2:sub, 3, :), [3 2 1]), [1, 59*(sub - 1)]));
[muhat_sub2_tst4,sigmahat_sub2_tst4] = normfit(reshape(permute(feature_tst(2:sub, 4, :), [3 2 1]), [1, 59*(sub - 1)]));
[muhat_sub2_tst5,sigmahat_sub2_tst5] = normfit(reshape(permute(feature_tst(2:sub, 5, :), [3 2 1]), [1, 59*(sub - 1)]));

%%% CDFs Based on Security Bits %%%
K_upbound = 100;
FAR = zeros(1, K_upbound*100);
for K = 0.01:0.01:K_upbound
    
    threshold = nthroot((10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]))^5/(2^(K + 1)),5);
    mu = [muhat_sub2_tst1 muhat_sub2_tst2 muhat_sub2_tst3 muhat_sub2_tst4 muhat_sub2_tst5];
    Sigma = cov([reshape(permute(feature(2:sub, 1, 181:239), [3 2 1]), [59*(sub - 1), 1]) reshape(permute(feature(2:sub, 2, 181:239), [3 2 1]), [59*(sub - 1), 1]) reshape(permute(feature(2:sub, 3, 181:239), [3 2 1]), [59*(sub - 1), 1])...
        reshape(permute(feature(2:sub, 4, 181:239), [3 2 1]), [59*(sub - 1), 1]) reshape(permute(feature(2:sub, 5, 181:239), [3 2 1]), [59*(sub - 1), 1])]);
    [F,err] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
        [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu,Sigma);
    FAR(1, int16(K*100)) = 100*F;
    
end

FRR = zeros(1, K_upbound*100);
for K = 0.01:0.01:K_upbound
    
    threshold = nthroot((10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]))^5/(2^(K + 1)),5);
    mu = [muhat_sub1_tst1 muhat_sub1_tst2 muhat_sub1_tst3 muhat_sub1_tst4 muhat_sub1_tst5];
    Sigma = cov([reshape(permute(feature(1, 1, 181:239), [2 1 3]), [59, 1]) reshape(permute(feature(1, 2, 181:239), [2 1 3]), [59, 1]) reshape(permute(feature(1, 3, 181:239), [2 1 3]), [59, 1])...
        reshape(permute(feature(1, 4, 181:239), [2 1 3]), [59, 1]) reshape(permute(feature(1, 5, 181:239), [2 1 3]), [59, 1])]);
    [F,err] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
        [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu,Sigma);
    FRR(1, int16(K*100)) = 100*(1 - F);
    
end

HTER = (FAR + FRR)/2;

K = 0.01:0.01:K_upbound;
plot(K, HTER);









