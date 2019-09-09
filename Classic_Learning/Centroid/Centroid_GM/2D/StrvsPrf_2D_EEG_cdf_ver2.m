%%%%%%%%%% Circle Integration Domain %%%%%%%%%%
clc, clear all

tic
%%% Data Prepration %%%
load 'EEG_Rest_Open.mat'

signal = zeros(106, 9600);
for i= 1:106
    % Channel F_PZ Data
    x = Physionet_EEG_MI(i,23,:);
    signal(i,:) = reshape (x, [1, 9600]);
end

%%% Feature Extraction %%%
feature = zeros(2, 2, 119);
for i= 1:2
    for j = 1:119     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, 1, j) = mean(signal_freq(8:13));
        feature(i, 2, j) = mean(signal_freq(14:30));
    end   
end

%%% Training Phase %%%%
[muhat_sub1_trn1,sigmahat_sub1_trn1] = normfit(reshape(feature(1, 1, 1:90), [1, 90]));
[muhat_sub1_trn2,sigmahat_sub1_trn2] = normfit(reshape(feature(1, 2, 1:90), [1, 90]));

[muhat_sub2_trn1,sigmahat_sub2_trn1] = normfit(reshape(feature(2, 1, 1:90), [1, 90]));
[muhat_sub2_trn2,sigmahat_sub2_trn2] = normfit(reshape(feature(2, 2, 1:90), [1, 90]));

%%% System Class %%%
[muhat_w_1,sigmahat_w_1] = normfit([reshape(feature(1, 1, 1:90), [1, 90]) reshape(feature(2, 1, 1:90), [1, 90])]);
[muhat_w_2,sigmahat_w_2] = normfit([reshape(feature(1, 2, 1:90), [1, 90]) reshape(feature(2, 2, 1:90), [1, 90])]);

%%% Testing Phase %%%
[muhat_sub1_tst1,sigmahat_sub1_tst1] = normfit(reshape(feature(1, 1, 91:119), [1, 29]));
[muhat_sub1_tst2,sigmahat_sub1_tst2] = normfit(reshape(feature(1, 2, 91:119), [1, 29]));

[muhat_sub2_tst1,sigmahat_sub2_tst1] = normfit(reshape(feature(2, 1, 91:119), [1, 29]));
[muhat_sub2_tst2,sigmahat_sub2_tst2] = normfit(reshape(feature(2, 2, 91:119), [1, 29]));

%%% CDFs %%%
thr = 0;
changing_seed = 1000;
FAR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);

iteration = 1000;

for i = 1:changing_seed
    i
    thr = thr + 3*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed; 
    mu = [muhat_sub2_tst1 muhat_sub2_tst2];
    Sigma = cov(reshape(feature(2, 1, 91:119), [29, 1]), reshape(feature(2, 2, 91:119), [29, 1]));
    
    %%% Integration over the Inner Square %%%
    [F_crc,err] = mvncdf([muhat_sub1_trn1 - thr*sqrt(2)/2 muhat_sub1_trn2 - thr*sqrt(2)/2],[muhat_sub1_trn1 + thr*sqrt(2)/2 muhat_sub1_trn2 + thr*sqrt(2)/2],mu,Sigma);
       
    for row = 1:iteration
        for col = 1:iteration

            point1 = [(col - 1) * (muhat_sub1_trn1 / iteration) (row - 1) * (muhat_sub1_trn2 / iteration)];
            point2 = [col * (muhat_sub1_trn1 / iteration) (row - 1) * (muhat_sub1_trn2 / iteration)];
            point3 = [col * (muhat_sub1_trn1 / iteration) row * (muhat_sub1_trn2 / iteration)];
            point4 = [(col - 1) * (muhat_sub1_trn1 / iteration) row * (muhat_sub1_trn2 / iteration)];

            point1_cond = (point1(1) - muhat_sub1_trn1)^2 + (point1(2) - muhat_sub1_trn2)^2;
            point2_cond = (point2(1) - muhat_sub1_trn1)^2 + (point2(2) - muhat_sub1_trn2)^2;
            point3_cond = (point3(1) - muhat_sub1_trn1)^2 + (point3(2) - muhat_sub1_trn2)^2;
            point4_cond = (point4(1) - muhat_sub1_trn1)^2 + (point4(2) - muhat_sub1_trn2)^2;
                      
            %%% (x - 0.5)^2 + (y - 0.5)^2 = 0.25 %%%
            if (point1_cond < thr^2) && (point2_cond < thr^2) && (point3_cond < thr^2) && (point4_cond < thr^2)
                if (point2(1) <= muhat_sub1_trn1 - thr*sqrt(2)/2) || (point1(1) >= muhat_sub1_trn1 + thr*sqrt(2)/2) || (point3(2) <= muhat_sub1_trn2 - thr*sqrt(2)/2) || (point1(2) >= muhat_sub1_trn2 + thr*sqrt(2)/2)
                    [F_sqr,err] = mvncdf([point1(1) point1(2)],[point3(1) point3(2)],mu,Sigma);
                    F_crc = F_crc + F_sqr;                             
                end
            end

        end
    end
    
    FAR(1, i) = 100*F_crc;
    security_bits(1, i) = log2(pi*(3*max(sigmahat_w_1, sigmahat_w_2))^2/(pi*thr^2));
    
end

thr = 0;
FRR = zeros(1, changing_seed);
for i = 1:changing_seed
    thr = thr + 3*max(sigmahat_sub1_trn1, sigmahat_sub1_trn2)/changing_seed; 
    mu = [muhat_sub1_tst1 muhat_sub1_tst2];
    Sigma = cov(reshape(feature(1, 1, 91:119), [29, 1]), reshape(feature(1, 2, 91:119), [29, 1]));
     
    %%% Integration over the Inner Square %%%
    [F_crc,err] = mvncdf([muhat_sub1_trn1 - thr*sqrt(2)/2 muhat_sub1_trn2 - thr*sqrt(2)/2],[muhat_sub1_trn1 + thr*sqrt(2)/2 muhat_sub1_trn2 + thr*sqrt(2)/2],mu,Sigma);
       
    for row = 1:iteration
        for col = 1:iteration

            point1 = [(col - 1) * (muhat_sub1_trn1 / iteration) (row - 1) * (muhat_sub1_trn2 / iteration)];
            point2 = [col * (muhat_sub1_trn1 / iteration) (row - 1) * (muhat_sub1_trn2 / iteration)];
            point3 = [col * (muhat_sub1_trn1 / iteration) row * (muhat_sub1_trn2 / iteration)];
            point4 = [(col - 1) * (muhat_sub1_trn1 / iteration) row * (muhat_sub1_trn2 / iteration)];

            point1_cond = (point1(1) - muhat_sub1_trn1)^2 + (point1(2) - muhat_sub1_trn2)^2;
            point2_cond = (point2(1) - muhat_sub1_trn1)^2 + (point2(2) - muhat_sub1_trn2)^2;
            point3_cond = (point3(1) - muhat_sub1_trn1)^2 + (point3(2) - muhat_sub1_trn2)^2;
            point4_cond = (point4(1) - muhat_sub1_trn1)^2 + (point4(2) - muhat_sub1_trn2)^2;
                      
            %%% (x - 0.5)^2 + (y - 0.5)^2 = 0.25 %%%
            if (point1_cond < thr^2) && (point2_cond < thr^2) && (point3_cond < thr^2) && (point4_cond < thr^2)
                if (point2(1) <= muhat_sub1_trn1 - thr*sqrt(2)/2) || (point1(1) >= muhat_sub1_trn1 + thr*sqrt(2)/2) || (point3(2) <= muhat_sub1_trn2 - thr*sqrt(2)/2) || (point1(2) >= muhat_sub1_trn2 + thr*sqrt(2)/2)
                    [F_sqr,err] = mvncdf([point1(1) point1(2)],[point3(1) point3(2)],mu,Sigma);
                    F_crc = F_crc + F_sqr;                             
                end
            end

        end
    end
    
    FRR(1, i) = 100*(1 - F_crc);
    
end

HTER = (FAR + FRR)/2;

figure
yyaxis left
% t1 = 1:1000;
% t2 = 1:20:1000;
% vq2 = interp1(t1,HTER,t2,'spline');
t3 = 0.003:3/changing_seed:3;
% plot(t2,vq2);
% plot(t3,vq2);
plot(t3,HTER);

% plot(HTER)
xlabel('Threshold');
ylabel('HTER(%)');

yyaxis right
t4 = 0.003:3/changing_seed:3;
plot(t4,security_bits)
ylabel('Security Strength (bits)');

toc







