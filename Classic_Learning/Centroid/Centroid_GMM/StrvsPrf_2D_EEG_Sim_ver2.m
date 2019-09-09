%%% Simulation over square subject space %%%%
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
feature = zeros(2, 2, 239);
for i= 1:2
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, 1, j) = mean(signal_freq(8:13));
        feature(i, 2, j) = mean(signal_freq(14:30));
    end   
end

%%% Data Plots %%%
% plot(reshape(feature(1, 1, 1:90), [1,90]), reshape(feature(1, 2, 1:90), [1,90]), '.b');
% hold on
% plot(reshape(feature(1, 1, 91:119), [1,29]), reshape(feature(1, 2, 91:119), [1,29]), '.b');
% hold on
% plot(reshape(feature(2, 1, 91:119), [1,29]), reshape(feature(2, 2, 91:119), [1,29]), '.r');

%%% Training Phase %%%%
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(feature(1, 1, 1:180), [1, 180]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(feature(1, 2, 1:180), [1, 180]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(feature(2, 1, 1:180), [1, 180]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(feature(2, 2, 1:180), [1, 180]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit([reshape(feature(1, 1, 1:180), [1, 180]) reshape(feature(2, 1, 1:180), [1, 180])]);
[muhat_w_2,sigmahat_w_2] = normfit([reshape(feature(1, 2, 1:180), [1, 180]) reshape(feature(2, 2, 1:180), [1, 180])]);

%%% Testing Phase %%%
[muhat_sub1_tst1,sigmahat_sub1_tst1] = normfit(reshape(feature(1, 1, 181:239), [1, 59]));
[muhat_sub1_tst2,sigmahat_sub1_tst2] = normfit(reshape(feature(1, 2, 181:239), [1, 59]));

[muhat_sub2_tst1,sigmahat_sub2_tst1] = normfit(reshape(feature(2, 1, 181:239), [1, 59]));
[muhat_sub2_tst2,sigmahat_sub2_tst2] = normfit(reshape(feature(2, 2, 181:239), [1, 59]));

% %%% Multivariate plot %%%
% mu = [muhat_sub1_tst1 muhat_sub1_tst2];
% Sigma = cov(reshape(feature(1, 1, 91:119), [29, 1]), reshape(feature(1, 2, 91:119), [29, 1]));
% x1 = -100:10:1200; x2 = -100:10:600;
% [X1,X2] = meshgrid(x1,x2);
% F = mvnpdf([X1(:) X2(:)],mu,Sigma);
% F = reshape(F,length(x2),length(x1));
% surf(x1,x2,F);
% caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
% axis([-100 1200 -100 600 0 0.00005])
% xlabel('x1'); ylabel('x2'); zlabel('Probability Density');

% %%% 2D Plot of Training Data %%%
% mu = [muhat_sub1_trn1 muhat_sub1_trn2];
% Sigma = cov(reshape(feature(1, 1, 1:90), [90, 1]), reshape(feature(1, 2, 1:90), [90, 1]));
% x1 = -100:10:1200; x2 = -100:10:600;
% [X1,X2] = meshgrid(x1,x2);
% F = mvnpdf([X1(:) X2(:)],mu,Sigma);
% F = reshape(F,length(x2),length(x1));
% 
% % mvncdf([0 0],[1 1],mu,Sigma);
% contour(x1,x2,F,[0:0.1e-5:1.2e-5]);
% xlabel('x'); ylabel('y');
% % line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');

% %%% 2D Plot of Subject Testing Data %%%
% mu = [muhat_sub1_tst1 muhat_sub1_tst2];
% Sigma = cov(reshape(feature(1, 1, 91:119), [29, 1]), reshape(feature(1, 2, 91:119), [29, 1]));
% x1 = -100:10:1200; x2 = -100:10:600;
% [X1,X2] = meshgrid(x1,x2);
% F = mvnpdf([X1(:) X2(:)],mu,Sigma);
% F = reshape(F,length(x2),length(x1));
% 
% % mvncdf([0 0],[1 1],mu,Sigma);
% contour(x1,x2,F,[0:0.3e-5:3.9e-5]);
% xlabel('x'); ylabel('y');
% % line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');

%%% 2D Plot of Subject Testing Data %%%
mu = [muhat_sub2_tst1 muhat_sub2_tst2];
Sigma = cov(reshape(feature(2, 1, 181:239), [59, 1]), reshape(feature(2, 2, 181:239), [59, 1]));
x1 = -100:10:1200; x2 = -100:10:600;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));

% mvncdf([0 0],[1 1],mu,Sigma);
% contour(x1,x2,F,[0:0.3e-5:3.9e-5]);
% xlabel('x'); ylabel('y');
% line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');

%%% Performance Simulation %%%
thr = 0;
changing_seed = 1000;
HTER = zeros(1, changing_seed);
for i = 1:changing_seed
    FR = 0;
    FA = 0;
    thr = thr + 5*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed;
    for j = 181:239
%         if ((feature(1,1,j) - muhat_sub1_trn1)^2 + (feature(1,2,j) - muhat_sub1_trn2)^2) >= thr^2
        if (feature(1,1,j) >= muhat_sub1_trn1 + thr) || (feature(1,1,j) <= muhat_sub1_trn1 - thr) || (feature(1,2,j) >= muhat_sub1_trn2 + thr) || (feature(1,2,j) <= muhat_sub1_trn2 - thr) 
            FR = FR + 1;
        end
%         if ((feature(2,1,j) - muhat_sub1_trn1)^2 + (feature(2,2,j) - muhat_sub1_trn2)^2) < thr^2
        if (feature(2,1,j) < muhat_sub1_trn1 + thr) && (feature(2,1,j) > muhat_sub1_trn1 - thr) && (feature(2,2,j) < muhat_sub1_trn2 + thr) && (feature(2,2,j) > muhat_sub1_trn2 - thr)
            FA = FA + 1;
        end
    end
    
    HTER(1, i) = 100*(FA/59 + FR/59)/2;
    
end

%%% Testing Phase for Attack Simulation %%%
security_bits = zeros(1, changing_seed);
thr = 0;
changing_seed = 1000;
for i = 1:changing_seed
    
    attempts = ones(1, 100);
    thr = thr + 5*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed;
    
    for j = 1:100
        rx = muhat_w_1 - 5*max(sigmahat_w_1, sigmahat_w_2) + (10*max(sigmahat_w_1, sigmahat_w_2)).*rand;
        ry = muhat_w_2 - 5*max(sigmahat_w_1, sigmahat_w_2) + (10*max(sigmahat_w_1, sigmahat_w_2)).*rand;
%         while ((rx - muhat_sub1_trn1)^2 + (ry - muhat_sub1_trn2)^2) >= thr^2
        while (muhat_sub1_trn1 - thr > rx) || (muhat_sub1_trn2 - thr > ry) || (muhat_sub1_trn1 + thr < rx) || (muhat_sub1_trn2 + thr < ry)
            attempts(1, j) = attempts(1, j) + 1;
            rx = muhat_w_1 - 5*max(sigmahat_w_1, sigmahat_w_2) + (10*max(sigmahat_w_1, sigmahat_w_2)).*rand;
            ry = muhat_w_2 - 5*max(sigmahat_w_1, sigmahat_w_2) + (10*max(sigmahat_w_1, sigmahat_w_2)).*rand;
        end
    end
    security_bits(1, i) = log2(mean(attempts(:)));
end

figure
yyaxis left
% t1 = 1:1000;
% t2 = 1:20:1000;
% vq2 = interp1(t1,HTER,t2,'spline');
t3 = 0.005:0.005:5;
% plot(t2,vq2);
% plot(t3,vq2);
plot(t3,HTER);

% plot(HTER)
xlabel('Threshold');
ylabel('HTER(%)');

yyaxis right
t4 = 0.005:0.005:5;
plot(t4,security_bits)
ylabel('Security Strength (bits)');







