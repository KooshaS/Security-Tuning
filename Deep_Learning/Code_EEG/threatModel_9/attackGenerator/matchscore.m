function y = matchscore(SVMModel, x)

    [~, s] = predict(SVMModel, x);
    y = - double(s(1))  % for generating 0, it should be '-'

end