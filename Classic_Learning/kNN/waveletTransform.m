% clc, clear all

function y = waveletTransform (signal)
    
    y = zeros(38, size(signal, 2));
    for j = 1:size(signal, 2)
        [cA1,cD1] = dwt(signal(:, j), 'sym4');
        [cA2,cD2] = dwt(cA1, 'sym4');
        [cA3,cD3] = dwt(cA2, 'sym4');
        [cA4,cD4] = dwt(cA3, 'sym4');                                    
        y(:, j) = cA4; 
    end
    
end

