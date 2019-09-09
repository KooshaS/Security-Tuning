%%%%%%%%% MI Dataset %%%%%%%%%
clc, clear all


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
[muhat_sub2_trn,sigmahat_sub2_trn] = normfit(reshape(permute(feature_trn(2:sub, 1:180), [2 1]), [1 180*(sub - 1)]));
[muhat_w,sigmahat_w] = normfit(reshape(permute(feature_trn(1:sub, 1:180), [2 1]), [1 180*sub]));

t = -500:1:1500;
norm_sub1 = normpdf(t,muhat_sub1_trn,sigmahat_sub1_trn);
% plot(t,norm_sub1,'b');
% hold on

norm_sub2 = normpdf(t,muhat_sub2_trn,sigmahat_sub2_trn);
% plot(t,norm_sub2,'r');

%%% Testing Phase Based on Security Bits %%%
K_upbound = 15;
HTER = zeros(1, K_upbound*100);

for K = 0.01:0.01:K_upbound
    FR = 0;
    FA = 0;
    threshold = 10*sigmahat_w/(2^(K + 1));
    for j = 181:239       
        if (feature(1,j) >= muhat_sub1_trn + threshold) || (feature(1,j) <= muhat_sub1_trn - threshold) 
            FR = FR + 1;
        end      
    end
    
    for k = 2:sub
        for j = 181:239       
            if (feature(k,j) < muhat_sub1_trn + threshold) && (feature(k,j) > muhat_sub1_trn - threshold)
                FA = FA + 1;
            end
        end
    end
    
    HTER(1, int16(K*100)) = 100*(FA/(59*(sub - 1)) + FR/59)/2;
end

K = 0.01:0.01:K_upbound;
plot(K, HTER);







