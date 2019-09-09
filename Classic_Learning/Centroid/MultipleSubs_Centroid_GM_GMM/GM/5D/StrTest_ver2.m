clc, clear all


%%% Data Prepration %%%
load 'Dataset1.mat'

signal = zeros(106, 19200);
for i= 1:106
    % Channel F_PZ Data
    x = Raw_Data(i,2,:);
    signal(i,:) = reshape (x, [1, 19200]);
end

%%% Feature Extraction %%%
feature = zeros(2, 1, 239);
for i= 1:2
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, 1, j) = mean(signal_freq(1:3));
%         feature(i, 2, j) = mean(signal_freq(4:7));
%         feature(i, 3, j) = mean(signal_freq(8:13));
%         feature(i, 4, j) = mean(signal_freq(14:30));
%         feature(i, 5, j) = mean(signal_freq(31:40));        
    end   
end

%%% Training Phase %%%%
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(feature(1, 1, 1:180), [1, 180]));
% [muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(feature(1, 2, 1:180), [1, 180]));
% [muhat_sub1_trn3,sigmahat_sub1_trn3] = normfit(reshape(feature(1, 3, 1:180), [1, 180]));
% [muhat_sub1_trn4,sigmahat_sub1_trn4] = normfit(reshape(feature(1, 4, 1:180), [1, 180]));
% [muhat_sub1_trn5,sigmahat_sub1_trn5] = normfit(reshape(feature(1, 5, 1:180), [1, 180]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(feature(2, 1, 1:180), [1, 180]));
% [muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(feature(2, 2, 1:180), [1, 180]));
% [muhat_sub2_trn3,sigmahat_sub2_trn3] = normfit(reshape(feature(2, 3, 1:180), [1, 180]));
% [muhat_sub2_trn4,sigmahat_sub2_trn4] = normfit(reshape(feature(2, 4, 1:180), [1, 180]));
% [muhat_sub2_trn5,sigmahat_sub2_trn5] = normfit(reshape(feature(2, 5, 1:180), [1, 180]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit([reshape(feature(1, 1, 1:180), [1, 180]) reshape(feature(2, 1, 1:180), [1, 180])]);
% [muhat_w_2,sigmahat_w_2] = normfit([reshape(feature(1, 2, 1:180), [1, 180]) reshape(feature(2, 2, 1:180), [1, 180])]);
% [muhat_w_3,sigmahat_w_3] = normfit([reshape(feature(1, 3, 1:180), [1, 180]) reshape(feature(2, 3, 1:180), [1, 180])]);
% [muhat_w_4,sigmahat_w_4] = normfit([reshape(feature(1, 4, 1:180), [1, 180]) reshape(feature(2, 4, 1:180), [1, 180])]);
% [muhat_w_5,sigmahat_w_5] = normfit([reshape(feature(1, 5, 1:180), [1, 180]) reshape(feature(2, 5, 1:180), [1, 180])]);

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
security_bits_t = zeros(1, changing_seed);
for i = 1:changing_seed
%     thr = thr + 3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed;
%     thr = thr + 3*mean([sigmahat_sub1_trn1 sigmahat_sub1_trn2])/changing_seed;
    thr = thr + 5*max([sigmahat_w_1])/changing_seed; 
%     security_bits_t(1, i) = log2(pi*(5*max(sigmahat_w_1, sigmahat_w_2))^2/(pi*thr^2));
%     security_bits_t(1, i) = log2(10*sigmahat_w_1*10*sigmahat_w_2*10*sigmahat_w_3*10*sigmahat_w_4*10*sigmahat_w_5/(2*thr)^5);
     security_bits_t(1, i) = log2((10*max([sigmahat_w_1]))^2/(2*thr)^2);
%     security_bits_t(1, i) = log2(10*sigmahat_w_1*10*sigmahat_w_2/(pi*thr)^2);
end

t = 0.003:0.003:3;
plot(t,security_bits_t)

%%% Testing Phase for Attack Simulation %%%
security_bits_e = zeros(1, changing_seed);
thr = 0;
% thr = 49*3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed;
% thr = 49*3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2])/changing_seed;
changing_seed = 1000;
for i = 1:changing_seed
    
    attempts = ones(1, 100);
%     thr = thr + 3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed;
% thr = thr + 3*mean([sigmahat_sub1_trn1 sigmahat_sub1_trn2])/changing_seed;
    thr = thr + 5*max([sigmahat_w_1])/changing_seed;
    
    for j = 1:100
        j
        rx1 = muhat_w_1 - 5*sigmahat_w_1 + (10*sigmahat_w_1).*rand;
%         rx2 = muhat_w_2 - 5*sigmahat_w_2 + (10*sigmahat_w_2).*rand;
%         rx3 = muhat_w_3 - 5*sigmahat_w_3 + (10*sigmahat_w_3).*rand;
%         rx4 = muhat_w_4 - 5*sigmahat_w_4 + (10*sigmahat_w_4).*rand;
%         rx5 = muhat_w_5 - 5*sigmahat_w_5 + (10*sigmahat_w_5).*rand;        
        
        cond_x1 = (rx1 >= muhat_sub1_trn1 + thr) || (rx1 <= muhat_sub1_trn1 - thr);
%         cond_x2 = (rx2 >= muhat_sub1_trn2 + thr) || (rx2 <= muhat_sub1_trn2 - thr);
%         cond_x3 = (rx3 >= muhat_sub1_trn3 + thr) || (rx3 <= muhat_sub1_trn3 - thr);
%         cond_x4 = (rx4 >= muhat_sub1_trn4 + thr) || (rx4 <= muhat_sub1_trn4 - thr);
%         cond_x5 = (rx5 >= muhat_sub1_trn5 + thr) || (rx5 <= muhat_sub1_trn5 - thr);       
        
        while (cond_x1)
%         while ((rx1 - muhat_sub1_trn1)^2 + (rx2 - muhat_sub1_trn2)^2) >= thr^2
            attempts(1, j) = attempts(1, j) + 1;
            rx1 = muhat_w_1 - 5*sigmahat_w_1 + (10*sigmahat_w_1).*rand;
%             rx2 = muhat_w_2 - 5*sigmahat_w_2 + (10*sigmahat_w_2).*rand;
%             rx3 = muhat_w_3 - 5*sigmahat_w_3 + (10*sigmahat_w_3).*rand;
%             rx4 = muhat_w_4 - 5*sigmahat_w_4 + (10*sigmahat_w_4).*rand;
%             rx5 = muhat_w_5 - 5*sigmahat_w_5 + (10*sigmahat_w_5).*rand;           
            
            cond_x1 = (rx1 >= muhat_sub1_trn1 + thr) || (rx1 <= muhat_sub1_trn1 - thr);
%             cond_x2 = (rx2 >= muhat_sub1_trn2 + thr) || (rx2 <= muhat_sub1_trn2 - thr);
%             cond_x3 = (rx3 >= muhat_sub1_trn3 + thr) || (rx3 <= muhat_sub1_trn3 - thr);
%             cond_x4 = (rx4 >= muhat_sub1_trn4 + thr) || (rx4 <= muhat_sub1_trn4 - thr);
%             cond_x5 = (rx5 >= muhat_sub1_trn5 + thr) || (rx5 <= muhat_sub1_trn5 - thr);            
        end
    end
    security_bits_e(1, i) = log2(mean(attempts(:)));
end

hold on
plot(t,security_bits_e)













