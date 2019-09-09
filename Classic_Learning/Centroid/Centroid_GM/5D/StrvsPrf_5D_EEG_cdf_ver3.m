%%%%%%%%%% Square Integration Domain Using MI %%%%%%%%%%
% clc, clear all


%%% Data Prepration %%%
load 'Dataset1.mat'

signal = zeros(106, 19200);
for i= 1:106
    % Channel F_PZ Data
    x = Raw_Data(i,2,:);
    signal(i,:) = reshape (x, [1, 19200]);
end

%%% Feature Extraction %%%
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
[muhat_sub1_tst1,sigmahat_sub1_tst1] = normfit(reshape(feature(1, 1, 181:239), [1, 59]));
[muhat_sub1_tst2,sigmahat_sub1_tst2] = normfit(reshape(feature(1, 2, 181:239), [1, 59]));
[muhat_sub1_tst3,sigmahat_sub1_tst3] = normfit(reshape(feature(1, 3, 181:239), [1, 59]));
[muhat_sub1_tst4,sigmahat_sub1_tst4] = normfit(reshape(feature(1, 4, 181:239), [1, 59]));
[muhat_sub1_tst5,sigmahat_sub1_tst5] = normfit(reshape(feature(1, 5, 181:239), [1, 59]));

[muhat_sub2_tst1,sigmahat_sub2_tst1] = normfit(reshape(feature(2, 1, 181:239), [1, 59]));
[muhat_sub2_tst2,sigmahat_sub2_tst2] = normfit(reshape(feature(2, 2, 181:239), [1, 59]));
[muhat_sub2_tst3,sigmahat_sub2_tst3] = normfit(reshape(feature(2, 3, 181:239), [1, 59]));
[muhat_sub2_tst4,sigmahat_sub2_tst4] = normfit(reshape(feature(2, 4, 181:239), [1, 59]));
[muhat_sub2_tst5,sigmahat_sub2_tst5] = normfit(reshape(feature(2, 5, 181:239), [1, 59]));

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
%     thr = thr + nthroot((15/(8*pi^2))*(10*sigmahat_w_1*10*sigmahat_w_2*10*sigmahat_w_3*10*sigmahat_w_4*10*sigmahat_w_5),5)/changing_seed;
%     thr = thr + 3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed;
    thr = thr + 5*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])/changing_seed;
    mu = [muhat_sub2_tst1 muhat_sub2_tst2 muhat_sub2_tst3 muhat_sub2_tst4 muhat_sub2_tst5];
    Sigma = cov([reshape(feature(2, 1, 181:239), [59, 1]) reshape(feature(2, 2, 181:239), [59, 1]) reshape(feature(2, 3, 181:239), [59, 1])...
        reshape(feature(2, 4, 181:239), [59, 1]) reshape(feature(2, 5, 181:239), [59, 1])]);
    [F,err] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr muhat_sub1_trn3 - thr muhat_sub1_trn4 - thr muhat_sub1_trn5 - thr],...
        [muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr muhat_sub1_trn3 + thr muhat_sub1_trn4 + thr muhat_sub1_trn5 + thr],mu,Sigma);
    FAR(1, i) = 100*F;
%     security_bits(1, i) = log2(pi*(3*max(sigmahat_w_1, sigmahat_w_2))^2/(pi*thr^2));
security_bits(1, i) = log2((10*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5]))^5/((2*thr)^5));
% security_bits(1, i) = log2(10*sigmahat_w_1*10*sigmahat_w_2*10*sigmahat_w_3*10*sigmahat_w_4*10*sigmahat_w_5/((8/15)*(pi^2)*thr^5));
% security_bits(1, i) = log2(10*sigmahat_w_1*10*sigmahat_w_2*10*sigmahat_w_3*10*sigmahat_w_4*10*sigmahat_w_5/(((pi^(5/2))/gamma((5/2)+1))*thr^5));
% security_bits(1, i) = log2(((((2/5)*(pi^(5/2))/gamma(5/2))*(5*sigmahat_w_1)*(5*sigmahat_w_2)*(5*sigmahat_w_3)*(5*sigmahat_w_4)*(5*sigmahat_w_5)))/(((pi^(5/2))/gamma((5/2)+1))*thr^5));
end

thr = 0;
changing_seed = 1000;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 5*max([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])/changing_seed;
%     thr = thr + nthroot((15/(8*pi^2))*(10*sigmahat_w_1*10*sigmahat_w_2*10*sigmahat_w_3*10*sigmahat_w_4*10*sigmahat_w_5),5)/changing_seed;
%     thr = thr + 3*max([sigmahat_sub1_trn1 sigmahat_sub1_trn2 sigmahat_sub1_trn3 sigmahat_sub1_trn4 sigmahat_sub1_trn5])/changing_seed;
%     thr = thr + 10*min([sigmahat_w_1 sigmahat_w_2 sigmahat_w_3 sigmahat_w_4 sigmahat_w_5])/changing_seed;
    mu = [muhat_sub1_tst1 muhat_sub1_tst2 muhat_sub1_tst3 muhat_sub1_tst4 muhat_sub1_tst5];
    Sigma = cov([reshape(feature(1, 1, 181:239), [59, 1]) reshape(feature(1, 2, 181:239), [59, 1]) reshape(feature(1, 3, 181:239), [59, 1])...
        reshape(feature(1, 4, 181:239), [59, 1]) reshape(feature(1, 5, 181:239), [59, 1])]);
    [F,err] = mvncdf([muhat_sub1_trn1 - thr muhat_sub1_trn2 - thr muhat_sub1_trn3 - thr muhat_sub1_trn4 - thr muhat_sub1_trn5 - thr],...
        [muhat_sub1_trn1 + thr muhat_sub1_trn2 + thr muhat_sub1_trn3 + thr muhat_sub1_trn4 + thr muhat_sub1_trn5 + thr],mu,Sigma);
    FRR(1, i) = 100*(1 - F);
end

HTER = (FAR + FRR)/2;

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







