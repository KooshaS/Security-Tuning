%%%%%%%%%% Square Integration Domain with K = 5 Components %%%%%%%%%%
clc, clear all


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
feature = zeros(2, 5, 239);
for i= 1:2
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
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(feature(1, 1, 1:180), [1, 180]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(feature(1, 2, 1:180), [1, 180]));
[muhat_sub1_trn3,sigmahat_sub1_trn3] = normfit(reshape(feature(1, 3, 1:180), [1, 180]));
[muhat_sub1_trn4,sigmahat_sub1_trn4] = normfit(reshape(feature(1, 4, 1:180), [1, 180]));
[muhat_sub1_trn5,sigmahat_sub1_trn5] = normfit(reshape(feature(1, 5, 1:180), [1, 180]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(feature(2, 1, 1:180), [1, 180]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(feature(2, 2, 1:180), [1, 180]));
[muhat_sub2_trn3,sigmahat_sub2_trn3] = normfit(reshape(feature(2, 3, 1:180), [1, 180]));
[muhat_sub2_trn4,sigmahat_sub2_trn4] = normfit(reshape(feature(2, 4, 1:180), [1, 180]));
[muhat_sub2_trn5,sigmahat_sub2_trn5] = normfit(reshape(feature(2, 5, 1:180), [1, 180]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit([reshape(feature(1, 1, 1:180), [1, 180]) reshape(feature(2, 1, 1:180), [1, 180])]);
[muhat_w_2,sigmahat_w_2] = normfit([reshape(feature(1, 2, 1:180), [1, 180]) reshape(feature(2, 2, 1:180), [1, 180])]);
[muhat_w_3,sigmahat_w_3] = normfit([reshape(feature(1, 3, 1:180), [1, 180]) reshape(feature(2, 3, 1:180), [1, 180])]);
[muhat_w_4,sigmahat_w_4] = normfit([reshape(feature(1, 4, 1:180), [1, 180]) reshape(feature(2, 4, 1:180), [1, 180])]);
[muhat_w_5,sigmahat_w_5] = normfit([reshape(feature(1, 5, 1:180), [1, 180]) reshape(feature(2, 5, 1:180), [1, 180])]);

%%% Testing Phase %%%
options = statset('Display','final');
GMModel_sub1_tst = fitgmdist(reshape(feature(1, :, 181:239), [5, 59])',5,'Options',options);
% hold on
% h = ezcontour(gmPDF,[0 1000],[0 600]);
% title('Scatter Plot and PDF Contour')
% hold off

GMModel_sub2_tst = fitgmdist(reshape(feature(2, :, 181:239), [5, 59])',5,'Options',options);
% hold on
% h = ezcontour(gmPDF,[0 1000],[0 600]);
% title('Scatter Plot and PDF Contour')
% hold off

%%% CDFs Based on Security Bits %%%
K_upbound = 100;
FAR = zeros(1, K_upbound*10);

for K = 0.1:0.1:K_upbound
    K
    threshold = nthroot((10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]))^5/(2^(K + 1)),5);
    mu1 = [GMModel_sub2_tst.mu(1,1) GMModel_sub2_tst.mu(1,2) GMModel_sub2_tst.mu(1,3) GMModel_sub2_tst.mu(1,4) GMModel_sub2_tst.mu(1,5)];
    Sigma1 = reshape(GMModel_sub2_tst.Sigma(:,:,1), [5, 5]);
    [F1,err1] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu1,Sigma1);
    
    mu2 = [GMModel_sub2_tst.mu(2,1) GMModel_sub2_tst.mu(2,2) GMModel_sub2_tst.mu(2,3) GMModel_sub2_tst.mu(2,4) GMModel_sub2_tst.mu(2,5)];
    Sigma2 = reshape(GMModel_sub2_tst.Sigma(:,:,2), [5, 5]);
    [F2,err2] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu2,Sigma2);
                   
    mu3 = [GMModel_sub2_tst.mu(3,1) GMModel_sub2_tst.mu(3,2) GMModel_sub2_tst.mu(3,3) GMModel_sub2_tst.mu(3,4) GMModel_sub2_tst.mu(3,5)];
    Sigma3 = reshape(GMModel_sub2_tst.Sigma(:,:,3), [5, 5]);
    [F3,err3] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu3,Sigma3);
                   
    mu4 = [GMModel_sub2_tst.mu(4,1) GMModel_sub2_tst.mu(4,2) GMModel_sub2_tst.mu(4,3) GMModel_sub2_tst.mu(4,4) GMModel_sub2_tst.mu(4,5)];
    Sigma4 = reshape(GMModel_sub2_tst.Sigma(:,:,4), [5, 5]);
    [F4,err4] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu4,Sigma4);
                   
    mu5 = [GMModel_sub2_tst.mu(5,1) GMModel_sub2_tst.mu(5,2) GMModel_sub2_tst.mu(5,3) GMModel_sub2_tst.mu(5,4) GMModel_sub2_tst.mu(5,5)];
    Sigma5 = reshape(GMModel_sub2_tst.Sigma(:,:,5), [5, 5]);
    [F5,err5] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu5,Sigma5);
    
    FAR(1, int16(K*10)) = 100*(GMModel_sub2_tst.ComponentProportion(1)*F1 + GMModel_sub2_tst.ComponentProportion(2)*F2 + GMModel_sub2_tst.ComponentProportion(3)*F3...
        + GMModel_sub2_tst.ComponentProportion(4)*F4 + GMModel_sub2_tst.ComponentProportion(5)*F5);
    
end

FRR = zeros(1, K_upbound*10);
for K = 0.1:0.1:K_upbound
    
    threshold = nthroot((10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]))^5/(2^(K + 1)),5);
    mu1 = [GMModel_sub1_tst.mu(1,1) GMModel_sub1_tst.mu(1,2) GMModel_sub1_tst.mu(1,3) GMModel_sub1_tst.mu(1,4) GMModel_sub1_tst.mu(1,5)];
    Sigma1 = reshape(GMModel_sub1_tst.Sigma(:,:,1), [5, 5]);
    [F1,err1] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu1,Sigma1);
    
    mu2 = [GMModel_sub1_tst.mu(2,1) GMModel_sub1_tst.mu(2,2) GMModel_sub1_tst.mu(2,3) GMModel_sub1_tst.mu(2,4) GMModel_sub1_tst.mu(2,5)];
    Sigma2 = reshape(GMModel_sub1_tst.Sigma(:,:,2), [5, 5]);
    [F2,err2] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu2,Sigma2);
                   
    mu3 = [GMModel_sub1_tst.mu(3,1) GMModel_sub1_tst.mu(3,2) GMModel_sub1_tst.mu(3,3) GMModel_sub1_tst.mu(3,4) GMModel_sub1_tst.mu(3,5)];
    Sigma3 = reshape(GMModel_sub1_tst.Sigma(:,:,3), [5, 5]);
    [F3,err3] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu3,Sigma3);
    
    mu4 = [GMModel_sub1_tst.mu(4,1) GMModel_sub1_tst.mu(4,2) GMModel_sub1_tst.mu(4,3) GMModel_sub1_tst.mu(4,4) GMModel_sub1_tst.mu(4,5)];
    Sigma1 = reshape(GMModel_sub1_tst.Sigma(:,:,4), [5, 5]);
    [F4,err4] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu4,Sigma4);
    
    mu5 = [GMModel_sub1_tst.mu(5,1) GMModel_sub1_tst.mu(5,2) GMModel_sub1_tst.mu(5,3) GMModel_sub1_tst.mu(5,4) GMModel_sub1_tst.mu(5,5)];
    Sigma5 = reshape(GMModel_sub1_tst.Sigma(:,:,5), [5, 5]);
    [F5,err5] = mvncdf([muhat_sub1_trn1 - threshold muhat_sub1_trn2 - threshold muhat_sub1_trn3 - threshold muhat_sub1_trn4 - threshold muhat_sub1_trn5 - threshold],...
                       [muhat_sub1_trn1 + threshold muhat_sub1_trn2 + threshold muhat_sub1_trn3 + threshold muhat_sub1_trn4 + threshold muhat_sub1_trn5 + threshold],mu5,Sigma5);
    
    FRR(1, int16(K*10)) = 100*(1 - GMModel_sub1_tst.ComponentProportion(1)*F1 - GMModel_sub1_tst.ComponentProportion(2)*F2 - GMModel_sub1_tst.ComponentProportion(3)*F3 ...
        - GMModel_sub1_tst.ComponentProportion(4)*F4 - GMModel_sub1_tst.ComponentProportion(5)*F5); 
    
end

HTER = (FAR + FRR)/2;

K = 0.1:0.1:K_upbound;
plot(K, HTER);






