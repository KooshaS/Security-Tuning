clc, clear all
%%%%%%%%% Rest Dataset %%%%%%%%%


%%% Data Prepration %%%
load 'EEG_Rest_Open.mat'

signal = zeros(106, 9600);
for i= 1:106
    % Channel F_PZ Data
    x = Physionet_EEG_MI(i,23,:);
    signal(i,:) = reshape (x, [1, 9600]);
end

%%% Feature Extraction %%%
feature = zeros(2, 119);
for i= 1:2
    for j = 1:119     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, j) = mean(signal_freq(8:13));
    end   
end

%%% Training Phase %%%%
[muhat_sub1_trn,sigmahat_sub1_trn] = normfit(feature(1, 1:90));
[muhat_sub2_trn,sigmahat_sub2_trn] = normfit(feature(2, 1:90));
[muhat_w,sigmahat_w] = normfit([feature(1, 1:90) feature(2, 1:90)]);

t = -500:1:1500;
norm_sub1 = normpdf(t,muhat_sub1_trn,sigmahat_sub1_trn);
plot(t,norm_sub1,'b');
hold on

norm_sub2 = normpdf(t,muhat_sub2_trn,sigmahat_sub2_trn);
plot(t,norm_sub2,'r');

%%% Testing Phase for HTER %%%
thr = 0;
changing_seed = 1000;
HTER = zeros(1, changing_seed);
for i = 1:changing_seed
    FR = 0;
    FA = 0;
    thr = thr + 3*sigmahat_sub1_trn/changing_seed;
    for j = 91:119       
        if (feature(1,j) >= muhat_sub1_trn + thr) || (feature(1,j) <= muhat_sub1_trn - thr) 
            FR = FR + 1;
        end
        if (feature(2,j) < muhat_sub1_trn + thr) && (feature(2,j) > muhat_sub1_trn - thr)
            FA = FA + 1;
        end
    end
    
    HTER(1, i) = 100*(FA/29 + FR/29)/2;
    
end

%%% Testing Phase for Attack Simulation %%%
security_bits = zeros(1, changing_seed);
thr = 0;
changing_seed = 1000;
for i = 1:changing_seed
    
    attempts = ones(1, 1000);
    thr = thr + 3*sigmahat_sub1_trn/changing_seed;
    
    for j = 1:1000
        r = muhat_w - 5*sigmahat_w + (10*sigmahat_w).*rand;
        while (r >= muhat_sub1_trn + thr) || (r <= muhat_sub1_trn - thr)
            attempts(1, j) = attempts(1, j) + 1;
            r = muhat_w - 5*sigmahat_w + (10*sigmahat_w).*rand;
        end
    end
    security_bits(1, i) = log2(mean(attempts(:)));
end

figure
yyaxis left
% t1 = 1:1000;
% t2 = 1:20:1000;
% vq2 = interp1(t1,HTER,t2,'spline');
t3 = 0.003:0.003:3;
% plot(t2,vq2);
% plot(t3,vq2);
plot(t3,HTER);

% plot(HTER)
xlabel('Threshold');
ylabel('HTER(%)');

yyaxis right
t4 = 0.003:0.003:3;
plot(t4,security_bits)
ylabel('Security Strength (bits)');







