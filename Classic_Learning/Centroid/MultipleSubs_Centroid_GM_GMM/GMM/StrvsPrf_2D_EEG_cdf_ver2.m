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
gmPDF2 = @(x,y) pdf(GMModel_sub2_tst,[x y]);
hold on
h = ezcontour(gmPDF2,[0 1000],[0 600], 120);
title('Scatter Plot and PDF Contour')
hold off

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 10*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed;
    
    mu1 = [GMModel_sub2_tst.mu(1,1) GMModel_sub2_tst.mu(1,2)];
    Sigma1 = reshape(GMModel_sub2_tst.Sigma(:,:,1), [2, 2]);
    [F1,err1] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu1,Sigma1);
    
    mu2 = [GMModel_sub2_tst.mu(2,1) GMModel_sub2_tst.mu(2,2)];
    Sigma2 = reshape(GMModel_sub2_tst.Sigma(:,:,2), [2, 2]);
    [F2,err2] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu2,Sigma2);
    
    mu3 = [GMModel_sub2_tst.mu(3,1) GMModel_sub2_tst.mu(3,2)];
    Sigma3 = reshape(GMModel_sub2_tst.Sigma(:,:,3), [2, 2]);
    [F3,err3] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu3,Sigma3);
    
%     mu4 = [GMModel_sub2_tst.mu(4,1) GMModel_sub2_tst.mu(4,2)];
%     Sigma4 = reshape(GMModel_sub2_tst.Sigma(:,:,4), [2, 2]);
%     [F4,err4] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu4,Sigma4);
%     
%     mu5 = [GMModel_sub2_tst.mu(5,1) GMModel_sub2_tst.mu(5,2)];
%     Sigma5 = reshape(GMModel_sub2_tst.Sigma(:,:,5), [2, 2]);
%     [F5,err5] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu5,Sigma5);
%     
%     mu6 = [GMModel_sub2_tst.mu(6,1) GMModel_sub2_tst.mu(6,2)];
%     Sigma6 = reshape(GMModel_sub2_tst.Sigma(:,:,6), [2, 2]);
%     [F6,err6] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu6,Sigma6);
%     
%     mu7 = [GMModel_sub2_tst.mu(7,1) GMModel_sub2_tst.mu(7,2)];
%     Sigma7 = reshape(GMModel_sub2_tst.Sigma(:,:,7), [2, 2]);
%     [F7,err7] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu7,Sigma7);
%     
%     mu8 = [GMModel_sub2_tst.mu(8,1) GMModel_sub2_tst.mu(8,2)];
%     Sigma8 = reshape(GMModel_sub2_tst.Sigma(:,:,8), [2, 2]);
%     [F8,err8] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu8,Sigma8);
%     
%     mu9 = [GMModel_sub2_tst.mu(9,1) GMModel_sub2_tst.mu(9,2)];
%     Sigma9 = reshape(GMModel_sub2_tst.Sigma(:,:,9), [2, 2]);
%     [F9,err9] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu9,Sigma9);
%     
%     mu10 = [GMModel_sub2_tst.mu(10,1) GMModel_sub2_tst.mu(10,2)];
%     Sigma10 = reshape(GMModel_sub2_tst.Sigma(:,:,10), [2, 2]);
%     [F10,err10] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu10,Sigma10);
    
%     FAR(1, i) = 100*(GMModel_sub2_tst.ComponentProportion(1)*F1 + GMModel_sub2_tst.ComponentProportion(2)*F2 + GMModel_sub2_tst.ComponentProportion(3)*F3...
%          + GMModel_sub2_tst.ComponentProportion(4)*F4 + GMModel_sub2_tst.ComponentProportion(5)*F5 + GMModel_sub2_tst.ComponentProportion(6)*F6...
%          + GMModel_sub2_tst.ComponentProportion(7)*F7);
     
     FAR(1, i) = 100*(GMModel_sub2_tst.ComponentProportion(1)*F1 + GMModel_sub2_tst.ComponentProportion(2)*F2 + GMModel_sub2_tst.ComponentProportion(3)*F3);

    security_bits(1, i) = log2((20*max(sigmahat_w_1, sigmahat_w_2))^2/(2*thr)^2);
end

thr = 0;
changing_seed = 1000;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 10*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed;
    
    mu1 = [GMModel_sub1_tst.mu(1,1) GMModel_sub1_tst.mu(1,2)];
    Sigma1 = reshape(GMModel_sub1_tst.Sigma(:,:,1), [2, 2]);
    [F1,err1] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu1,Sigma1);
    
    mu2 = [GMModel_sub1_tst.mu(2,1) GMModel_sub1_tst.mu(2,2)];
    Sigma2 = reshape(GMModel_sub1_tst.Sigma(:,:,2), [2, 2]);
    [F2,err2] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu2,Sigma2);
    
    mu3 = [GMModel_sub1_tst.mu(3,1) GMModel_sub1_tst.mu(3,2)];
    Sigma3 = reshape(GMModel_sub1_tst.Sigma(:,:,3), [2, 2]);
    [F3,err3] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu3,Sigma3);
    
%     mu4 = [GMModel_sub1_tst.mu(4,1) GMModel_sub1_tst.mu(4,2)];
%     Sigma4 = reshape(GMModel_sub1_tst.Sigma(:,:,4), [2, 2]);
%     [F4,err4] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu4,Sigma4);
%     
%     mu5 = [GMModel_sub1_tst.mu(5,1) GMModel_sub1_tst.mu(5,2)];
%     Sigma5 = reshape(GMModel_sub1_tst.Sigma(:,:,5), [2, 2]);
%     [F5,err5] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu5,Sigma5);
%     
%     mu6 = [GMModel_sub1_tst.mu(6,1) GMModel_sub1_tst.mu(6,2)];
%     Sigma6 = reshape(GMModel_sub1_tst.Sigma(:,:,6), [2, 2]);
%     [F6,err6] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu6,Sigma6);
%     
%     mu7 = [GMModel_sub1_tst.mu(7,1) GMModel_sub1_tst.mu(7,2)];
%     Sigma7 = reshape(GMModel_sub1_tst.Sigma(:,:,7), [2, 2]);
%     [F7,err7] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu7,Sigma7);
%     
%     mu8 = [GMModel_sub1_tst.mu(8,1) GMModel_sub1_tst.mu(8,2)];
%     Sigma8 = reshape(GMModel_sub1_tst.Sigma(:,:,8), [2, 2]);
%     [F8,err8] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu8,Sigma8);
%     
%     mu9 = [GMModel_sub1_tst.mu(9,1) GMModel_sub1_tst.mu(9,2)];
%     Sigma9 = reshape(GMModel_sub1_tst.Sigma(:,:,9), [2, 2]);
%     [F9,err9] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu9,Sigma9);
%     
%     mu10 = [GMModel_sub1_tst.mu(10,1) GMModel_sub1_tst.mu(10,2)];
%     Sigma10 = reshape(GMModel_sub1_tst.Sigma(:,:,10), [2, 2]);
%     [F10,err10] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr],[muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr],mu10,Sigma10);
    
%     FRR(1, i) = 100*(1 - GMModel_sub1_tst.ComponentProportion(1)*F1 - GMModel_sub1_tst.ComponentProportion(2)*F2 - GMModel_sub1_tst.ComponentProportion(3)*F3...
%         - GMModel_sub1_tst.ComponentProportion(4)*F4 - GMModel_sub1_tst.ComponentProportion(5)*F5 - GMModel_sub1_tst.ComponentProportion(6)*F6...
%         - GMModel_sub1_tst.ComponentProportion(7)*F7);
    
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
xlabel('Threshold');
ylabel('HTER(%)');

yyaxis right
t4 = 0.01:0.01:10;
plot(t4,security_bits)
ylabel('Security Strength (bits)');







