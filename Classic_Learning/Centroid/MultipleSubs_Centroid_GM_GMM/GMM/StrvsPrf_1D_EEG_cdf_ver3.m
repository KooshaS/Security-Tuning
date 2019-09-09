%%%%%%%%%% Using Gaussian Mixture Distribution for MI data modeling %%%%%%%%%%
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

%%% Testing Phase %%%
feature_tst = feature(:, 181:239);
GMModel_sub1_tst = fitgmdist(feature(1, 181:239)',3);
GMModel_sub2_tst = fitgmdist(reshape(permute(feature_tst(2:sub, :), [2 1]), [1 59*(sub - 1)])',3);

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 10*sigmahat_sub1_trn/changing_seed;
    
    mu1 = GMModel_sub2_tst.mu(1,1);
    Sigma1 = reshape(GMModel_sub2_tst.Sigma(:,:,1), [1, 1]);
    [F1,err1] = mvncdf([muhat_sub1_trn - thr],[muhat_sub1_trn + thr],mu1,Sigma1);
    
    mu2 = GMModel_sub2_tst.mu(2,1);
    Sigma2 = reshape(GMModel_sub2_tst.Sigma(:,:,2), [1, 1]);
    [F2,err2] = mvncdf([muhat_sub1_trn - thr],[muhat_sub1_trn + thr],mu2,Sigma2);
    
    mu3 = GMModel_sub2_tst.mu(3,1);
    Sigma3 = reshape(GMModel_sub2_tst.Sigma(:,:,3), [1, 1]);
    [F3,err3] = mvncdf([muhat_sub1_trn - thr],[muhat_sub1_trn + thr],mu3,Sigma3);
    
    FAR(1, i) = 100*(GMModel_sub2_tst.ComponentProportion(1)*F1 + GMModel_sub2_tst.ComponentProportion(2)*F2 + GMModel_sub2_tst.ComponentProportion(3)*F3);
    
    security_bits(1, i) = log2(20*sigmahat_w/(2*thr));
end

thr = 0;
changing_seed = 1000;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 10*sigmahat_sub1_trn/changing_seed;
    
    mu1 = GMModel_sub1_tst.mu(1,1);
    Sigma1 = reshape(GMModel_sub1_tst.Sigma(:,:,1), [1, 1]);
    [F1,err1] = mvncdf([muhat_sub1_trn - thr],[muhat_sub1_trn + thr],mu1,Sigma1);
    
    mu2 = GMModel_sub1_tst.mu(2,1);
    Sigma2 = reshape(GMModel_sub1_tst.Sigma(:,:,2), [1, 1]);
    [F2,err2] = mvncdf([muhat_sub1_trn - thr],[muhat_sub1_trn + thr],mu2,Sigma2);
    
    mu3 = GMModel_sub1_tst.mu(3,1);
    Sigma3 = reshape(GMModel_sub1_tst.Sigma(:,:,3), [1, 1]);
    [F3,err3] = mvncdf([muhat_sub1_trn - thr],[muhat_sub1_trn + thr],mu3,Sigma3);
    
    FRR(1, i) = 100*(1 - GMModel_sub1_tst.ComponentProportion(1)*F1 - GMModel_sub1_tst.ComponentProportion(2)*F2 - GMModel_sub1_tst.ComponentProportion(3)*F3);
    
end

HTER = (FAR + FRR)/2;

figure
yyaxis left
% t1 = 1:1000;
% t2 = 1:20:1000;
% vq2 = interp1(t1,HTER,t2,'spline');
t3 = 0.01:0.01:10;
% plot(t2,vq2);
% plot(t3,vq2);
plot(t3,HTER);

% plot(HTER)
xlabel('Threshold (\tau)');
ylabel('HTER(%)');

yyaxis right
t4 = 0.01:0.01:10;
plot(t4,security_bits)
ylabel('Security Strength (bits)');










