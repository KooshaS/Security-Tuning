
maximum1 = zeros(1, 100);
maximum2 = zeros(1, 100);
l = zeros(1, 100);
for lambda = 0.01:0.01:1
    
    product = lambda .* accuracy(1:160) - (1 - lambda) .* attSuccess(1:160); 
    x = lambda * 100;
    l(int16(x)) = lambda
    maximum1(1, int16(x)) = max(product(:));
    maximum2(1, int16(x)) = min(find(product == max(product(:))));
    
end

product = 0.5 .* accuracy(1:160) - (1 - 0.5) .* attSuccess(1:160);
figure
plot(product)
figure
plot(l, maximum2)


