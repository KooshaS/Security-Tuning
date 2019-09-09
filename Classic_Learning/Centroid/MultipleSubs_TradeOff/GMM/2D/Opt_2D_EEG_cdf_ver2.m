%%%%%%%%%% Square Integration Domain with K = 3 Components %%%%%%%%%%
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
feature = zeros(sub, 2, 239);
for i= 1:sub
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, 1, j) = mean(signal_freq(8:13));
        feature(i, 2, j) = mean(signal_freq(14:30));
    end   
end

%%% Training Phase %%%%
feature_trn = feature(:, :, 1:180);
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(permute(feature_trn(1, 1, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(permute(feature_trn(1, 2, 1:180), [2 1 3]), [1, 180]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(permute(feature_trn(2:sub, 1, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(permute(feature_trn(2:sub, 2, 1:180), [3 2 1]), [1, 180*(sub - 1)]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit(reshape(permute(feature_trn(1:sub, 1, 1:180), [3 2 1]), [1, 180*sub]));
[muhat_w_2,sigmahat_w_2] = normfit(reshape(permute(feature_trn(1:sub, 2, 1:180), [3 2 1]), [1, 180*sub]));

%%% Testing Phase %%%
options = statset('Display','final');
feature_tst = feature(:, :, 181:239);
GMModel_sub1_tst = fitgmdist(reshape(permute(feature_tst(1, :, :), [2 1 3]), [2, 59])',3,'Options',options);
% gmPDF1 = @(x,y)pdf(GMModel_sub1_tst,[x y]);
% hold on
% h = ezcontour(gmPDF1,[200 1000],[0 600], 120);
% title('Scatter Plot and PDF Contour')
% hold off

GMModel_sub2_tst = fitgmdist(reshape(permute(feature_tst(2:sub, :, :), [2 3 1]), [2, 59*(sub - 1)])',3,'Options',options);
% % % gmPDF2 = @(x,y)pdf(GMModel_sub2_tst,[x y]);
% % % hold on
% % % h = ezcontour(gmPDF2,[0 1000],[0 600], 120);
% % % title('Scatter Plot and PDF Contour')
% % % hold off

%%% CDFs Based on Security Bits %%%
K_upbound = 25;
FAR = zeros(1, K_upbound*100);

for K = 0.01:0.01:K_upbound
    threshold = sqrt((10*sigmahat_w_1*10*sigmahat_w_2)/(pi*2^K));
      
    mu1 = [GMModel_sub2_tst.mu(1,1) GMModel_sub2_tst.mu(1,2)];
    Sigma1 = reshape(GMModel_sub2_tst.Sigma(:,:,1), [2, 2]);
    [F1,err1] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu1,Sigma1);
    
    mu2 = [GMModel_sub2_tst.mu(2,1) GMModel_sub2_tst.mu(2,2)];
    Sigma2 = reshape(GMModel_sub2_tst.Sigma(:,:,2), [2, 2]);
    [F2,err2] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu2,Sigma2);
    
    mu3 = [GMModel_sub2_tst.mu(3,1) GMModel_sub2_tst.mu(3,2)];
    Sigma3 = reshape(GMModel_sub2_tst.Sigma(:,:,3), [2, 2]);
    [F3,err3] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu3,Sigma3);
    
    FAR(1, int16(K*100)) = 100*(GMModel_sub2_tst.ComponentProportion(1)*F1 + GMModel_sub2_tst.ComponentProportion(2)*F2 + GMModel_sub2_tst.ComponentProportion(3)*F3);
    
end

FRR = zeros(1, K_upbound*100);
for K = 0.01:0.01:K_upbound
    threshold = sqrt((10*sigmahat_w_1*10*sigmahat_w_2)/(pi*2^K));
    
    mu1 = [GMModel_sub1_tst.mu(1,1) GMModel_sub1_tst.mu(1,2)];
    Sigma1 = reshape(GMModel_sub1_tst.Sigma(:,:,1), [2, 2]);
    [F1,err1] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu1,Sigma1);
    
    mu2 = [GMModel_sub1_tst.mu(2,1) GMModel_sub1_tst.mu(2,2)];
    Sigma2 = reshape(GMModel_sub1_tst.Sigma(:,:,2), [2, 2]);
    [F2,err2] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu2,Sigma2);
    
    mu3 = [GMModel_sub1_tst.mu(3,1) GMModel_sub1_tst.mu(3,2)];
    Sigma3 = reshape(GMModel_sub1_tst.Sigma(:,:,3), [2, 2]);
    [F3,err3] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold],[muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold],mu3,Sigma3);
    
    FRR(1, int16(K*100)) = 100*(1 - GMModel_sub1_tst.ComponentProportion(1)*F1 - GMModel_sub1_tst.ComponentProportion(2)*F2 - GMModel_sub1_tst.ComponentProportion(3)*F3); 
    
end

HTER = (FAR + FRR)/2;

K = 0.01:0.01:K_upbound;
plot(K, HTER);






