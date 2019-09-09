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

%%% Data Plots %%%
% plot(reshape(feature(1, 1, 1:90), [1,90]), reshape(feature(1, 2, 1:90), [1,90]), '.b');
% hold on
% plot(reshape(feature(1, 1, 91:119), [1,29]), reshape(feature(1, 2, 91:119), [1,29]), '.b');
% hold on
% plot(reshape(feature(2, 1, 91:119), [1,29]), reshape(feature(2, 2, 91:119), [1,29]), '.r');

%%% Training Phase %%%%
feature_trn = feature(:, :, 1:180);
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(permute(feature_trn(1, 1, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(permute(feature_trn(1, 2, 1:180), [2 1 3]), [1, 180]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(permute(feature_trn(2:end, 1, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(permute(feature_trn(2:end, 2, 1:180), [3 2 1]), [1, 180*(sub - 1)]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit(reshape(permute(feature_trn(1:sub, 1, 1:180), [3 2 1]), [1, 180*sub]));
[muhat_w_2,sigmahat_w_2] = normfit(reshape(permute(feature_trn(1:sub, 2, 1:180), [3 2 1]), [1, 180*sub]));

%%% Testing Phase %%%
feature_tst = feature(:, :, 181:239);
[muhat_sub1_tst1,sigmahat_sub1_tst1] = normfit(reshape(permute(feature_tst(1, 1, :), [2 1 3]), [1, 59]));
[muhat_sub1_tst2,sigmahat_sub1_tst2] = normfit(reshape(permute(feature_tst(1, 2, :), [2 1 3]), [1, 59]));

[muhat_sub2_tst1,sigmahat_sub2_tst1] = normfit(reshape(permute(feature_tst(2:sub, 1, :), [3 2 1]), [1, 59*(sub - 1)]));
[muhat_sub2_tst2,sigmahat_sub2_tst2] = normfit(reshape(permute(feature_tst(2:sub, 2, :), [3 2 1]), [1, 59*(sub - 1)]));

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
Sigma = cov(reshape(permute(feature(2:sub, 1, 181:239), [3 2 1]), [59*(sub - 1), 1]), reshape(permute(feature(2:sub, 2, 181:239), [3 2 1]), [59*(sub - 1), 1]));
x1 = -100:10:1200; x2 = -100:10:600;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));

% mvncdf([0 0],[1 1],mu,Sigma);
contour(x1,x2,F,[0:0.3e-5:3.9e-5]);
xlabel('x'); ylabel('y');
% line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');

%%% Performance Simulation %%%
%%% Testing Phase Based on Security Bits %%%
K_upbound = 25;
HTER = zeros(1, K_upbound*100);

for K = 0.01:0.01:K_upbound
    FR = 0;
    FA = 0;
    threshold = sqrt((10*sigmahat_w_1*10*sigmahat_w_2)/(pi*2^K));  
    for j = 181:239
%         if ((feature(1,1,j) - muhat_sub1_trn1)^2 + (feature(1,2,j) - muhat_sub1_trn2)^2) >= threshold^2
        if (feature(1,1,j) >= muhat_sub1_trn1 + threshold) || (feature(1,1,j) <= muhat_sub1_trn1 - threshold) || (feature(1,2,j) >= muhat_sub1_trn2 + threshold) || (feature(1,2,j) <= muhat_sub1_trn2 - threshold) 
            FR = FR + 1;
        end
    end
    
    for k = 2:sub
        for j = 181:239        
%             if ((feature(k,1,j) - muhat_sub1_trn1)^2 + (feature(k,2,j) - muhat_sub1_trn2)^2) < threshold^2
            if (feature(k,1,j) < muhat_sub1_trn1 + threshold) && (feature(k,1,j) > muhat_sub1_trn1 - threshold) && (feature(k,2,j) < muhat_sub1_trn2 + threshold) && (feature(k,2,j) > muhat_sub1_trn2 - threshold)            
                FA = FA + 1;
            end
        end
    end
    
    HTER(1, int16(K*100)) = 100*(FA/(59*(sub - 1)) + FR/59)/2;
end

K = 0.01:0.01:K_upbound;
plot(K, HTER);









