%%%%%%%%% MI Dataset %%%%%%%%%
% clc, clear all


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
feature = zeros(sub, 239);
for i= 1:sub
    for j = 1:239     
        signal_freq = abs(fft(signal(i, (j - 1)*80 + 1:(j - 1)*80 + 160)));
        signal_freq = signal_freq(2:160/2+1);                    
        feature(i, j) = mean(signal_freq(8:13));
    end   
end

%%% Training Phase %%%%
feature_trn = feature(:, 1:180);
[muhat_sub1_trn,sigmahat_sub1_trn] = normfit(feature(1, 1:180));
[muhat_sub2_trn,sigmahat_sub2_trn] = normfit(reshape(permute(feature_trn(2:sub, :), [2 1]), [1, 180*(sub - 1)]));
[muhat_w,sigmahat_w] = normfit(reshape(permute(feature_trn(1:sub, :), [2 1]), [1, 180*sub]));

t = -500:1:1500;
norm_sub1 = normpdf(t,muhat_sub1_trn,sigmahat_sub1_trn);
plot(t,norm_sub1,'b');
hold on

norm_sub2 = normpdf(t,muhat_sub2_trn,sigmahat_sub2_trn);
plot(t,norm_sub2,'r');

%%% Testing Phase %%%
thr = 0;
changing_seed = 1000;
HTER = zeros(1, changing_seed);
CCR = zeros(1, changing_seed);
security_bits = zeros(1, changing_seed);
for i = 1:changing_seed
    FR = 0;
    FA = 0;
    thr = thr + 10*sigmahat_sub1_trn/changing_seed;
    for j = 181:239       
        if (feature(1,j) >= muhat_sub1_trn + thr) || (feature(1,j) <= muhat_sub1_trn - thr) 
            FR = FR + 1;
        end        
    end
    
    for k = 2:sub
        for j = 181:239        
            if (feature(k,j) < muhat_sub1_trn + thr) && (feature(k,j) > muhat_sub1_trn - thr)
                FA = FA + 1;
            end
        end
    end
    
    HTER(1, i) = 100*(FA/(59*(sub - 1)) + FR/59)/2;
%     CCR(1, i) = 100*(((89 - FA) + (89 - FR))/178);
    security_bits(1, i) = log2(20*sigmahat_w/(2*thr));
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







