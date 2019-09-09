clc, clear all


load ovariancancer.mat
obs(:, 2:end) = [];

X = [obs(1:90,:); obs(122:211,:)+0.08];

%%% Training Phase %%%
[muhat_c1,sigmahat_c1] = normfit(X(1:50, :));
[muhat_c2,sigmahat_c2] = normfit(X(91:140, :));
[muhat_w,sigmahat_w] = normfit(X(1:180, :));

t = -0.4:.001:0.4;
norm_c1 = normpdf(t,muhat_c1,sigmahat_c1);
plot(t,norm_c1,'b');
hold on

norm_c2 = normpdf(t,muhat_c2,sigmahat_c2);
plot(t,norm_c2,'r');

norm_w = normpdf(t,muhat_w,sigmahat_w);
plot(t,norm_w);

%%% Testing Phase %%%
thr = 6*sigmahat_c1;
changing_seed = 100;
HTER = zeros(1, changing_seed);
for i = 1:changing_seed
    FA = 0;
    FR = 0;
    for j = 1:40       
        if (X(50+j,:) > muhat_c1 + thr) || (X(50+j,:) < muhat_c1 - thr) 
            FR = FR + 1;
        elseif (X(140+j,:) < muhat_c1 + thr) && (X(140+j,:) > muhat_c1 - thr)
            FA = FA + 1;
        end
    end
    thr = thr - 6*sigmahat_c1/changing_seed;
    HTER(1, i) = 100*(FA/30 + FR/30)/2;
end

figure
plot(HTER)










