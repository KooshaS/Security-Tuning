%%% Simulation over square subject space %%%%
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
[muhat_sub1_trn3,sigmahat_sub1_trn3] = normfit(reshape(permute(feature_trn(1, 3, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn4,sigmahat_sub1_trn4] = normfit(reshape(permute(feature_trn(1, 4, 1:180), [2 1 3]), [1, 180]));
[muhat_sub1_trn5,sigmahat_sub1_trn5] = normfit(reshape(permute(feature_trn(1, 5, 1:180), [2 1 3]), [1, 180]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(permute(feature_trn(2:end, 1, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(permute(feature_trn(2:end, 2, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn3,sigmahat_sub2_trn3] = normfit(reshape(permute(feature_trn(2:end, 3, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn4,sigmahat_sub2_trn4] = normfit(reshape(permute(feature_trn(2:end, 4, 1:180), [3 2 1]), [1, 180*(sub - 1)]));
[muhat_sub2_trn5,sigmahat_sub2_trn5] = normfit(reshape(permute(feature_trn(2:end, 5, 1:180), [3 2 1]), [1, 180*(sub - 1)]));

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
% mu = [muhat_sub2_tst1 muhat_sub2_tst2];
% Sigma = cov(reshape(feature(2, 1, 181:239), [59, 1]), reshape(feature(2, 2, 181:239), [59, 1]));
% x1 = -100:10:1200; x2 = -100:10:600;
% [X1,X2] = meshgrid(x1,x2);
% F = mvnpdf([X1(:) X2(:)],mu,Sigma);
% F = reshape(F,length(x2),length(x1));

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
    thr = thr + 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])/changing_seed;
%     thr = thr + 3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed; 
    for j = 181:239
%         if ((feature(1,1,j) - muhat_sub1_trn1)^2 + (feature(1,2,j) - muhat_sub1_trn2)^2) >= thr^2
        
        cond_x1 = (feature(1,1,j) >= muhat_sub1_trn1 + thr) || (feature(1,1,j) <= muhat_sub1_trn1 - thr);
        cond_x2 = (feature(1,2,j) >= muhat_sub1_trn2 + thr) || (feature(1,2,j) <= muhat_sub1_trn2 - thr);
        cond_x3 = (feature(1,3,j) >= muhat_sub1_trn3 + thr) || (feature(1,3,j) <= muhat_sub1_trn3 - thr);
        cond_x4 = (feature(1,4,j) >= muhat_sub1_trn4 + thr) || (feature(1,4,j) <= muhat_sub1_trn4 - thr);
        cond_x5 = (feature(1,5,j) >= muhat_sub1_trn5 + thr) || (feature(1,5,j) <= muhat_sub1_trn5 - thr);        
        
        if (cond_x1 || cond_x2 || cond_x3 || cond_x4 || cond_x5)
            FR = FR + 1;
        end
    end
    
    for k = 2:sub
        for j = 181:239
%         if ((feature(2,1,j) - muhat_sub1_trn1)^2 + (feature(2,2,j) - muhat_sub1_trn2)^2) < thr^2

            cond_x1 = (feature(k,1,j) < muhat_sub1_trn1 + thr) && (feature(k,1,j) > muhat_sub1_trn1 - thr);
            cond_x2 = (feature(k,2,j) < muhat_sub1_trn2 + thr) && (feature(k,2,j) > muhat_sub1_trn2 - thr);
            cond_x3 = (feature(k,3,j) < muhat_sub1_trn3 + thr) && (feature(k,3,j) > muhat_sub1_trn3 - thr);
            cond_x4 = (feature(k,4,j) < muhat_sub1_trn4 + thr) && (feature(k,4,j) > muhat_sub1_trn4 - thr);
            cond_x5 = (feature(k,5,j) < muhat_sub1_trn5 + thr) && (feature(k,5,j) > muhat_sub1_trn5 - thr);        

            if (cond_x1 && cond_x2 && cond_x3 && cond_x4 && cond_x5)
                FA = FA + 1;
            end
        end
    end
    
    HTER(1, i) = 100*(FA/(59*(sub - 1)) + FR/59)/2;
    
end

%%% Testing Phase for Attack Simulation %%%
security_bits = zeros(1, changing_seed);
thr = 0;
thr = thr + 199*10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])/changing_seed;
% thr = 49*3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed;
changing_seed = 1000;
for i = 200:changing_seed
    
    attempts = ones(1, 100);
    thr = thr + 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])/changing_seed;
%     thr = thr + 3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed;
    
    for j = 1:100
        j
        rx1 = muhat_w_1 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
        rx2 = muhat_w_2 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
        rx3 = muhat_w_3 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
        rx4 = muhat_w_4 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
        rx5 = muhat_w_5 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;         
        
        cond_x1 = (rx1 >= muhat_sub1_trn1 + thr) || (rx1 <= muhat_sub1_trn1 - thr);
        cond_x2 = (rx2 >= muhat_sub1_trn2 + thr) || (rx2 <= muhat_sub1_trn2 - thr);
        cond_x3 = (rx3 >= muhat_sub1_trn3 + thr) || (rx3 <= muhat_sub1_trn3 - thr);
        cond_x4 = (rx4 >= muhat_sub1_trn4 + thr) || (rx4 <= muhat_sub1_trn4 - thr);
        cond_x5 = (rx5 >= muhat_sub1_trn5 + thr) || (rx5 <= muhat_sub1_trn5 - thr);        
        
        while (cond_x1 || cond_x2 || cond_x3 || cond_x4 || cond_x5)
%         while ((rx1 - muhat_sub1_trn1)^2 + (rx2 - muhat_sub1_trn2)^2 + (rx3 - muhat_sub1_trn3)^2 + (rx4 - muhat_sub1_trn4)^2 + (rx5 - muhat_sub1_trn5)^2) >= thr^2
            attempts(1, j) = attempts(1, j) + 1;
            rx1 = muhat_w_1 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
            rx2 = muhat_w_2 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
            rx3 = muhat_w_3 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
            rx4 = muhat_w_4 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand;
            rx5 = muhat_w_5 - 10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]) + (20*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])).*rand; 
            
            cond_x1 = (rx1 >= muhat_sub1_trn1 + thr) || (rx1 <= muhat_sub1_trn1 - thr);
            cond_x2 = (rx2 >= muhat_sub1_trn2 + thr) || (rx2 <= muhat_sub1_trn2 - thr);
            cond_x3 = (rx3 >= muhat_sub1_trn3 + thr) || (rx3 <= muhat_sub1_trn3 - thr);
            cond_x4 = (rx4 >= muhat_sub1_trn4 + thr) || (rx4 <= muhat_sub1_trn4 - thr);
            cond_x5 = (rx5 >= muhat_sub1_trn5 + thr) || (rx5 <= muhat_sub1_trn5 - thr);            
        end
    end
    security_bits(1, i) = log2(mean(attempts(:)));
end

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







