clc, clear all


point1 = 0;
point2 = 0;
point3 = 0;
point4 = 0;
volume = 0;

iteration = 1000;
latency = zeros(1, iteration);
est_err = zeros(1, iteration);

row_granularity = 0;
col_granularity = 0;

for i = 1:iteration
    i
    tic
    volume = 0;
    row_granularity = row_granularity + 1;
    col_granularity = col_granularity + 1;
    
    for row = 1:row_granularity
        for col = 1:col_granularity

            point1 = [(col - 1) * (1 / col_granularity) (row - 1) * (1 / row_granularity)];
            point2 = [col * (1 / col_granularity) (row - 1) * (1 / row_granularity)];
            point3 = [col * (1 / col_granularity) row * (1 / row_granularity)];
            point4 = [(col - 1) * (1 / col_granularity) row * (1 / row_granularity)];

            point1_cond = (point1(1) - 0.5)^2 + (point1(2) - 0.5)^2;
            point2_cond = (point2(1) - 0.5)^2 + (point2(2) - 0.5)^2;
            point3_cond = (point3(1) - 0.5)^2 + (point3(2) - 0.5)^2;
            point4_cond = (point4(1) - 0.5)^2 + (point4(2) - 0.5)^2;
                      
            %%% (x - 0.5)^2 + (y - 0.5)^2 = 0.25 %%%
            if (point1_cond < 0.25) && (point2_cond < 0.25) && (point3_cond < 0.25) && (point4_cond < 0.25)
                volume = volume + (1 / col_granularity) * (1 / row_granularity);
            end

        end
    end
    
    latency(1, i) = toc;
    est_err(1, i) = 100 * (pi*(0.5)^2 - volume)/(pi*(0.5)^2);

end

%%% Plots %%%
figure
yyaxis left
t1 = 1:iteration;
plot(t1,est_err);

% plot(HTER)
xlabel('Square Root of Number of Pixels');
ylabel('Estimation Error (%)');

yyaxis right
t2 = 1:iteration;
plot(t2,latency)
ylabel('Computation Time (s)');






