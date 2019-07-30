%%% Classifying Using 100 Subjects Data Using Fixed Training %%%
function [acc, mdl] = kNNFunc(xData_trn, yData_trn, xData_tst, yData_tst, N)

% for N = 5:5
%     mdl = fitcknn(xData_trn, yData_trn, 'NumNeighbors', N, 'Standardize', 1);
    mdl = fitcknn(xData_trn, yData_trn, 'NumNeighbors',...
        N, 'Standardize', 1, 'Distance', 'spearman');
    [label, score, cost] = predict(mdl, xData_tst);
    acc = 100 * sum(label == yData_tst)/numel(yData_tst);
% end

end


