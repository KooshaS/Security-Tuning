


random = optimizationResults(cell2mat(optimizationResults(:, 8)) > 75, :);

random(101:end, :) = [];

find(cell2mat(random(:, end)) == max(cell2mat(random(:, end))));

random2 = sort(cell2mat(random));

